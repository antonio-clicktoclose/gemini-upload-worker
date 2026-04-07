/**
 * Scoring Worker - EasyPanel
 * 
 * worker.ts - Polls pgmq for scoring jobs, processes them with Gemini + Claude/OpenAI
 * 
 * FIXES APPLIED:
 * 1. extractJSON: bulletproof JSON extraction (strips preamble, fences, headers, repairs truncation)
 * 2. Token limits set to 16384 for all providers
 * 3. Retry logic with stronger prompt enforcement on parse failure
 * 4. Prompt hardening to prevent markdown output
 */

import fs from 'fs';
import https from 'https';
import http from 'http';
import {
  uploadFileToGemini,
  waitForFileActive,
  deleteGeminiFile,
  analyzeAudioWithGemini,
} from './scoring.js';

const SUPABASE_URL = process.env.SUPABASE_URL!;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const POLL_INTERVAL_MS = 5_000;
const MAX_RETRIES = 3;
const VISIBILITY_TIMEOUT = 300; // seconds

// ============================================================
// BULLETPROOF JSON EXTRACTION
// ============================================================

function extractJSON(raw: string): any {
  if (!raw || !raw.trim()) throw new Error('Empty AI response');

  let text = raw;

  // 1. Strip markdown code fences
  text = text.replace(/^```(?:json)?\s*\n?/im, '').replace(/\n?\s*```\s*$/im, '');

  // 2. Strip everything before the first { or [
  const firstBrace = text.indexOf('{');
  const firstBracket = text.indexOf('[');
  let startIdx = -1;
  if (firstBrace === -1 && firstBracket === -1) {
    throw new Error(`No JSON object found in AI response. First 300 chars: ${raw.substring(0, 300)}`);
  } else if (firstBrace === -1) {
    startIdx = firstBracket;
  } else if (firstBracket === -1) {
    startIdx = firstBrace;
  } else {
    startIdx = Math.min(firstBrace, firstBracket);
  }
  text = text.substring(startIdx);

  // 3. Strip everything after the last matching } or ]
  const lastBrace = text.lastIndexOf('}');
  const lastBracket = text.lastIndexOf(']');
  const endIdx = Math.max(lastBrace, lastBracket);
  if (endIdx === -1) {
    throw new Error(`No closing brace/bracket found. First 300 chars: ${text.substring(0, 300)}`);
  }
  text = text.substring(0, endIdx + 1);

  // 4. Remove control characters (except \n and \t)
  text = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, '');

  // 5. Fix trailing commas before } or ]
  text = text.replace(/,\s*([}\]])/g, '$1');

  // 6. First parse attempt
  try {
    return JSON.parse(text);
  } catch (_) {
    // continue to repair
  }

  // 7. Repair: balance unclosed quotes, brackets, braces
  console.log('[extractJSON] First parse failed, attempting repair...');
  let repaired = text;
  const quoteCount = (repaired.match(/(?<!\\)"/g) || []).length;
  if (quoteCount % 2 !== 0) repaired += '"';
  const openBrackets = (repaired.match(/\[/g) || []).length;
  const closeBrackets = (repaired.match(/]/g) || []).length;
  const openBraces = (repaired.match(/{/g) || []).length;
  const closeBraces = (repaired.match(/}/g) || []).length;
  repaired = repaired.replace(/,\s*$/, '');
  repaired += ']'.repeat(Math.max(0, openBrackets - closeBrackets));
  repaired += '}'.repeat(Math.max(0, openBraces - closeBraces));

  // 8. Second parse attempt
  try {
    const result = JSON.parse(repaired);
    console.log('[extractJSON] Repair successful');
    return result;
  } catch (e) {
    console.error('[extractJSON] Repair failed. First 500 chars:', text.substring(0, 500));
    throw new Error(`Failed to parse AI response as JSON after repair. First 300 chars: ${raw.substring(0, 300)}`);
  }
}

// ============================================================
// SUPABASE RPC HELPERS
// ============================================================

async function supabaseFetch(path: string, options: any = {}) {
  const res = await fetch(`${SUPABASE_URL}${path}`, {
    ...options,
    headers: {
      apikey: SUPABASE_SERVICE_KEY,
      Authorization: `Bearer ${SUPABASE_SERVICE_KEY}`,
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  return res;
}

async function pgmqRead(): Promise<any | null> {
  const res = await supabaseFetch('/rest/v1/rpc/pgmq_read', {
    method: 'POST',
    body: JSON.stringify({
      queue_name: 'scoring_jobs',
      vt: VISIBILITY_TIMEOUT,
      qty: 1,
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    console.error('pgmq_read error:', text);
    return null;
  }

  const rows = await res.json() as any[];
  return rows?.[0] || null;
}

async function pgmqDelete(msgId: number): Promise<void> {
  await supabaseFetch('/rest/v1/rpc/pgmq_delete', {
    method: 'POST',
    body: JSON.stringify({ queue_name: 'scoring_jobs', msg_id: msgId }),
  });
}

async function pgmqArchive(msgId: number): Promise<void> {
  await supabaseFetch('/rest/v1/rpc/pgmq_archive', {
    method: 'POST',
    body: JSON.stringify({ queue_name: 'scoring_jobs', msg_id: msgId }),
  });
}

async function updateCallScore(scoreId: string, updates: Record<string, any>): Promise<void> {
  const res = await supabaseFetch(`/rest/v1/call_scores?id=eq.${scoreId}`, {
    method: 'PATCH',
    headers: { Prefer: 'return=minimal' },
    body: JSON.stringify({ ...updates, updated_at: new Date().toISOString() }),
  });
  if (!res.ok) {
    console.error('Failed to update call_scores:', await res.text());
  }
}

async function insertScoreDetails(details: any[]): Promise<void> {
  const res = await supabaseFetch('/rest/v1/call_score_details', {
    method: 'POST',
    headers: { Prefer: 'return=minimal' },
    body: JSON.stringify(details),
  });
  if (!res.ok) {
    console.error('Failed to insert score details:', await res.text());
  }
}

async function deleteScoreDetails(scoreId: string): Promise<void> {
  await supabaseFetch(`/rest/v1/call_score_details?call_score_id=eq.${scoreId}`, {
    method: 'DELETE',
  });
}

// ============================================================
// DOWNLOAD RECORDING TO DISK
// ============================================================

function downloadFile(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    const client = url.startsWith('https') ? https : http;

    const request = client.get(url, { timeout: 300_000 }, (response) => {
      // Follow redirects
      if (response.statusCode && response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        file.close();
        fs.unlinkSync(dest);
        return downloadFile(response.headers.location, dest).then(resolve).catch(reject);
      }

      if (response.statusCode !== 200) {
        file.close();
        fs.unlinkSync(dest);
        return reject(new Error(`Download failed: ${response.statusCode}`));
      }

      let downloaded = 0;
      response.on('data', (chunk: Buffer) => {
        downloaded += chunk.length;
      });

      response.pipe(file);
      file.on('finish', () => {
        file.close();
        console.log(`Downloaded ${(downloaded / 1024 / 1024).toFixed(1)}MB to ${dest}`);
        resolve();
      });
    });

    request.on('error', (err) => {
      file.close();
      if (fs.existsSync(dest)) fs.unlinkSync(dest);
      reject(err);
    });

    request.on('timeout', () => {
      request.destroy();
      reject(new Error('Download timeout'));
    });
  });
}

// ============================================================
// PROCESS A SINGLE SCORING JOB
// ============================================================

async function processJob(job: any): Promise<void> {
  const {
    call_id,
    sub_account_id,
    workspace_id,
    call_score_id,
    recording_url,
    transcript,
    transcript_source,
    recording_id,
    rubric,
    call_context,
    scored_by,
    gemini_api_key,
    gemini_model,
    scoring_api_key,
    scoring_model,
    scoring_provider,
  } = job;

  console.log(`Processing job for call ${call_id} (score ${call_score_id})`);

  const tmpPath = `/tmp/scoring-${Date.now() % 10000}.mp4`;
  let geminiFileName: string | null = null;

  try {
    // 1. Download recording to disk
    await downloadFile(recording_url, tmpPath);

    // 2. Upload to Gemini
    const mimeType = recording_url.match(/\.(mp3|wav|m4a|ogg|webm)(\?|$)/i) ? 'audio/mp4' : 'video/mp4';
    const uploaded = await uploadFileToGemini(tmpPath, gemini_api_key, mimeType);
    geminiFileName = uploaded.name;

    // 3. Wait for ACTIVE
    const isActive = await waitForFileActive(uploaded.name, gemini_api_key);

    let audioDelivery: any = null;
    let analysisType = 'transcript_only';

    if (isActive) {
      // 4. Analyze with Gemini
      try {
        const audioPrompt = buildAudioAnalysisPrompt(call_context);
        const audioResult = await analyzeAudioWithGemini(
          uploaded.uri,
          mimeType,
          audioPrompt,
          gemini_api_key,
          gemini_model || 'gemini-2.5-flash'
        );
        audioDelivery = extractJSON(audioResult);
        analysisType = 'audio_and_transcript';
        console.log('Audio analysis complete');
      } catch (audioErr) {
        console.error('Audio analysis failed:', audioErr);
        // Fall through to transcript-only scoring
      }
    } else {
      console.error('Gemini file never became ACTIVE');
    }

    // 5. Score transcript with retry logic
    const scoringResult = await scoreTranscriptWithRetry(
      transcript,
      rubric,
      call_context,
      audioDelivery,
      scoring_api_key,
      scoring_model,
      scoring_provider
    );

    // 6. Merge audio delivery into result
    const enrichedAudioDelivery = audioDelivery
      ? {
          ...audioDelivery,
          delivery_grade: audioDelivery.delivery_grade || null,
          delivery_summary: audioDelivery.delivery_summary || null,
        }
      : null;

    const maxPossible = scoringResult.max_possible_score || rubric.max_total_score;
    const scorePercentage = maxPossible > 0
      ? Math.round((scoringResult.overall_score / maxPossible) * 100)
      : 0;

    // 7. Update call_scores
    await updateCallScore(call_score_id, {
      overall_score: scoringResult.overall_score,
      max_possible_score: maxPossible,
      analysis_type: analysisType,
      audio_delivery: enrichedAudioDelivery,
      objections_detected: scoringResult.objections_detected || [],
      coaching_notes: scoringResult.coaching_notes || '',
      strengths: scoringResult.strengths || [],
      improvements: scoringResult.improvements || [],
      transcript_source,
      recording_id,
      ai_provider: scoring_provider,
      ai_model: scoring_model,
      scored_by,
      status: 'completed',
      scored_at: new Date().toISOString(),
      error_message: null,
    });

    // 8. Insert step details
    if (scoringResult.step_scores?.length) {
      await deleteScoreDetails(call_score_id);
      const details = scoringResult.step_scores.map((step: any) => ({
        call_score_id,
        step_key: step.step_key,
        step_label: step.step_label,
        score: step.score || 0,
        max_score: step.max_score || 0,
        reasoning: step.reasoning || '',
        evidence: step.evidence || [],
        adherence_percentage: step.adherence_percentage ?? null,
        key_phrases_hit: step.key_phrases_hit ?? null,
        key_phrases_missed: step.key_phrases_missed ?? null,
      }));
      await insertScoreDetails(details);
    }

    // 9. Send in-app notification for the rep
    try {
      await sendRepNotification(
        call_id, sub_account_id, call_score_id,
        call_context, scorePercentage, scoringResult
      );
    } catch (notifErr) {
      console.error('Notification failed (non-blocking):', notifErr);
    }

    // 10. Trigger Slack alert
    try {
      await supabaseFetch('/functions/v1/send-slack-alert', {
        method: 'POST',
        body: JSON.stringify({
          sub_account_id,
          alert_type: 'call_scored',
          data: {
            rep_name: call_context.rep_name,
            contact_name: call_context.contact_name,
            overall_score: scoringResult.overall_score,
            max_possible_score: maxPossible,
            score_percentage: scorePercentage,
            framework_name: rubric.name,
            analysis_type: analysisType,
            step_scores: (scoringResult.step_scores || []).map((s: any) => ({
              label: s.step_label,
              score: s.score || 0,
              max: s.max_score || 0,
            })),
            delivery_grade: audioDelivery?.delivery_grade || null,
            delivery_summary: audioDelivery?.delivery_summary || null,
            deal_killer: scoringResult.deal_killer || null,
            coaching_fixes: scoringResult.coaching_fixes || [],
            assigned_drill: scoringResult.assigned_drill || null,
          },
        }),
      });
    } catch (slackErr) {
      console.error('Slack alert failed (non-blocking):', slackErr);
    }

    console.log(`Call ${call_id} scored: ${scorePercentage}% (${scoring_provider}/${scoring_model})`);

  } finally {
    // Cleanup
    if (fs.existsSync(tmpPath)) {
      fs.unlinkSync(tmpPath);
      console.log(`Cleaned up: ${tmpPath}`);
    }
    if (geminiFileName) {
      await deleteGeminiFile(geminiFileName, gemini_api_key);
    }
  }
}

// ============================================================
// SEND IN-APP NOTIFICATION TO REP
// ============================================================

async function sendRepNotification(
  callId: string,
  subAccountId: string,
  callScoreId: string,
  callContext: any,
  scorePercentage: number,
  scoringResult: any
): Promise<void> {
  const repEmail = callContext.rep_email;
  if (!repEmail) return;

  // Look up rep's profile
  const res = await supabaseFetch(
    `/rest/v1/profiles?email=eq.${encodeURIComponent(repEmail)}&select=id&limit=1`
  );
  if (!res.ok) return;
  const profiles = await res.json() as any[];
  if (!profiles?.[0]?.id) return;

  const grade = scorePercentage >= 90 ? 'A' : scorePercentage >= 80 ? 'B' : scorePercentage >= 70 ? 'C' : scorePercentage >= 60 ? 'D' : 'F';
  const severity = scorePercentage >= 80 ? 'success' : scorePercentage >= 50 ? 'warning' : 'error';
  const dealKiller = scoringResult.deal_killer;
  const topFix = (scoringResult.coaching_fixes || [])[0];
  const takeaway = dealKiller ? `Deal killer: ${dealKiller}` : topFix ? `Top fix: ${topFix}` : '';
  const messageBody = takeaway ? `${takeaway} | call_id:${callId}` : `call_id:${callId}`;

  await supabaseFetch('/rest/v1/ai_insight_notifications', {
    method: 'POST',
    headers: { Prefer: 'return=minimal' },
    body: JSON.stringify({
      user_id: profiles[0].id,
      sub_account_id: subAccountId,
      insight_type: 'call_scored',
      insight_hash: `call_scored_${callScoreId}`,
      title: `Call Scored: ${callContext.contact_name || 'Unknown'} — ${scorePercentage}% (${grade})`,
      message: messageBody,
      severity,
      notification_channel: 'in_app',
      sent_at: new Date().toISOString(),
    }),
  });

  console.log(`Notification sent to rep ${repEmail} for call ${callId}`);
}

// ============================================================
// SCORE TRANSCRIPT WITH RETRY
// ============================================================

const MAX_SCORING_ATTEMPTS = 2;

async function scoreTranscriptWithRetry(
  transcript: string,
  rubric: any,
  callContext: any,
  audioDelivery: any | null,
  apiKey: string,
  model: string,
  provider: string
): Promise<any> {
  const systemPrompt = buildScoringSystemPrompt(rubric, callContext, audioDelivery);
  const baseUserPrompt = `Score this sales call transcript:\n\n${transcript}`;

  for (let attempt = 1; attempt <= MAX_SCORING_ATTEMPTS; attempt++) {
    try {
      const promptSuffix = attempt > 1
        ? '\n\nCRITICAL: Your previous response was not valid JSON. Return ONLY a raw JSON object. No markdown, no headers, no code fences, no explanatory text. Start with { and end with }.'
        : '';

      const userPrompt = baseUserPrompt + promptSuffix;
      let response: string;

      if (provider === 'anthropic') {
        response = await callAnthropic(apiKey, model, systemPrompt, userPrompt);
      } else if (provider === 'openai') {
        response = await callOpenAI(apiKey, model, systemPrompt, userPrompt);
      } else {
        response = await callOpenRouter(apiKey, model, systemPrompt, userPrompt);
      }

      return extractJSON(response);
    } catch (e) {
      console.error(`Scoring attempt ${attempt}/${MAX_SCORING_ATTEMPTS} failed:`, e instanceof Error ? e.message : String(e));
      if (attempt === MAX_SCORING_ATTEMPTS) {
        throw e;
      }
      console.log('Retrying with stronger JSON enforcement...');
    }
  }

  throw new Error('Scoring failed after all attempts');
}

async function callAnthropic(apiKey: string, model: string, system: string, user: string): Promise<string> {
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      max_tokens: 16384,
      system,
      messages: [{ role: 'user', content: user }],
    }),
  });

  if (!res.ok) throw new Error(`Anthropic error: ${res.status} ${await res.text()}`);
  const data = await res.json() as any;
  return data.content?.[0]?.text || '';
}

async function callOpenAI(apiKey: string, model: string, system: string, user: string): Promise<string> {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      max_tokens: 16384,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
    }),
  });

  if (!res.ok) throw new Error(`OpenAI error: ${res.status} ${await res.text()}`);
  const data = await res.json() as any;
  return data.choices?.[0]?.message?.content || '';
}

async function callOpenRouter(apiKey: string, model: string, system: string, user: string): Promise<string> {
  const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      max_tokens: 16384,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
    }),
  });

  if (!res.ok) throw new Error(`OpenRouter error: ${res.status} ${await res.text()}`);
  const data = await res.json() as any;
  return data.choices?.[0]?.message?.content || '';
}

// ============================================================
// PROMPT BUILDERS
// ============================================================

function buildAudioAnalysisPrompt(callContext: any): string {
  return `Analyze this sales call recording for delivery quality. The rep is ${callContext.rep_name || 'unknown'} and the prospect is ${callContext.contact_name || 'unknown'}.

Return a JSON object with these fields:
{
  "delivery_grade": "A-F letter grade",
  "delivery_summary": "one sentence summary of delivery quality",
  "tone": "description of tone",
  "energy_level": "low/medium/high",
  "vocal_confidence": "description",
  "pacing": "description",
  "enthusiasm_level": "low/medium/high",
  "rapport_quality": "description",
  "voice_variation": "description",
  "talk_to_listen_ratio": "e.g. 60:40",
  "filler_word_count": number,
  "strategic_pauses": number,
  "interruption_count": number,
  "emotional_shifts": [{"timestamp": "MM:SS", "from_emotion": "", "to_emotion": "", "trigger": ""}],
  "critical_moments": [{"timestamp": "MM:SS", "description": "", "impact": "positive/negative"}],
  "visual_analysis": {"eye_contact": "", "environment": "", "body_language": ""},
  "engagement_timeline": [{"timestamp": "MM:SS", "engagement_level": 1-10, "note": ""}],
  "active_listening_signals": [""],
  "monologue_flags": [{"timestamp": "MM:SS", "duration_seconds": number, "speaker": ""}]
}

CRITICAL OUTPUT FORMAT: Return ONLY the raw JSON object. Do NOT include markdown headers, code fences, or any text before or after the JSON.`;
}

function buildScoringSystemPrompt(rubric: any, callContext: any, audioDelivery: any | null): string {
  const steps = rubric.steps || [];
  const stepsDescription = steps.map((s: any) =>
    `- ${s.key} (${s.label}, max ${s.max_points} pts): ${s.description}\n  Criteria: ${s.criteria.join(', ')}`
  ).join('\n');

  const audioContext = audioDelivery
    ? `\n\nAudio/Video Delivery Analysis Available:\n${JSON.stringify(audioDelivery, null, 2)}\n\nUse this delivery data to inform your scoring, especially for confidence, engagement, and objection handling.`
    : '';

  return `You are a sales call scoring expert. Score this call using the "${rubric.name}" framework.

Rep: ${callContext.rep_name || 'Unknown'}
Prospect: ${callContext.contact_name || 'Unknown'}
Outcome: ${callContext.outcome_type || 'Unknown'}

Scoring Steps:
${stepsDescription}
${audioContext}

Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{
  "overall_score": number,
  "max_possible_score": ${rubric.max_total_score},
  "step_scores": [
    {
      "step_key": "string",
      "step_label": "string",
      "score": number,
      "max_score": number,
      "reasoning": "string (2-3 sentences)",
      "evidence": ["direct quote or timestamp"],
      "adherence_percentage": number (0-100),
      "key_phrases_hit": ["string"],
      "key_phrases_missed": ["string"]
    }
  ],
  "objections_detected": [
    {
      "objection": "string",
      "handled_well": boolean,
      "response_quality": "excellent|good|fair|poor",
      "evidence": "string"
    }
  ],
  "strengths": ["string"],
  "improvements": ["string"],
  "coaching_notes": "string",
  "deal_killer": "string or null",
  "coaching_fixes": ["string"],
  "assigned_drill": "string or null"
}

CRITICAL OUTPUT FORMAT: Return ONLY the raw JSON object. Do NOT include markdown headers, code fences, explanatory text, or any content before or after the JSON. Your entire response must be valid JSON starting with { and ending with }.`;
}

// ============================================================
// POLL LOOP
// ============================================================

async function pollLoop(): Promise<void> {
  console.log('Scoring worker started, polling for jobs...');

  while (true) {
    try {
      const msg = await pgmqRead();

      if (msg) {
        const job = msg.message;
        const msgId = msg.msg_id;
        const readCount = msg.read_ct || 1;

        try {
          await processJob(job);
          await pgmqArchive(msgId);
          console.log(`Job ${msgId} completed and archived`);
        } catch (jobErr: any) {
          console.error(`Job ${msgId} failed (attempt ${readCount}):`, jobErr);

          if (readCount >= MAX_RETRIES) {
            // Mark as failed and archive
            await updateCallScore(job.call_score_id, {
              status: 'failed',
              error_message: `Worker failed after ${MAX_RETRIES} attempts: ${jobErr.message}`,
            });
            await pgmqArchive(msgId);
            console.error(`Job ${msgId} permanently failed after ${MAX_RETRIES} attempts`);
          }
          // Otherwise, let visibility timeout expire so it retries
        }
      }
    } catch (pollErr) {
      console.error('Poll error:', pollErr);
    }

    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
}

// ============================================================
// HEALTH CHECK + STARTUP
// ============================================================

import express from 'express';

const app = express();
const PORT = process.env.PORT || 3000;

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', uptime: process.uptime() });
});

app.listen(PORT, () => {
  console.log(`Health server on port ${PORT}`);
  pollLoop().catch((err) => {
    console.error('Poll loop crashed:', err);
    process.exit(1);
  });
});
