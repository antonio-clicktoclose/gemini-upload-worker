import * as fs from 'node:fs';
import * as http from 'node:http';
import * as https from 'node:https';
import {
  analyzeAudioWithGemini,
  deleteGeminiFile,
  estimateGeminiTokens,
  exceedsGeminiMediaTokenLimit,
  extractAudioToOgg,
  getMediaDurationSeconds,
  uploadFileToGemini,
  waitForFileActive,
} from './scoring';

const SUPABASE_URL = process.env.SUPABASE_URL ?? '';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY ?? '';
const POLL_INTERVAL_MS = 5_000;
const MAX_RETRIES = 3;
const VISIBILITY_TIMEOUT = 300;

type MediaAnalysisMode = 'video_and_audio' | 'audio_only' | 'transcript_only';
type AnalysisType = 'audio_and_transcript' | 'audio_only' | 'transcript_only';

type QueueMessage = {
  msg_id: number;
  read_ct: number;
  enqueued_at: string;
  vt: string;
  message: Record<string, any>;
  headers?: Record<string, any>;
};

type AnthropicResponse = {
  content?: Array<{ text?: string }>;
};

type OpenAIResponse = {
  choices?: Array<{ message?: { content?: string } }>;
};

type OpenRouterResponse = {
  choices?: Array<{ message?: { content?: string } }>;
};

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  throw new Error('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required');
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeMediaMode(value: unknown): MediaAnalysisMode {
  if (value === 'audio_only' || value === 'transcript_only' || value === 'video_and_audio') {
    return value;
  }
  return 'video_and_audio';
}

function mapModeToAnalysisType(mode: MediaAnalysisMode): AnalysisType {
  if (mode === 'audio_only') return 'audio_only';
  if (mode === 'video_and_audio') return 'audio_and_transcript';
  return 'transcript_only';
}

function inferExtensionFromUrl(url: string): string {
  try {
    const pathname = new URL(url).pathname.toLowerCase();
    const match = pathname.match(/\.([a-z0-9]+)$/i);
    return match?.[1] || 'mp4';
  } catch {
    const match = url.toLowerCase().match(/\.([a-z0-9]+)(?:\?|$)/i);
    return match?.[1] || 'mp4';
  }
}

function inferMimeType(extension: string): { mimeType: string; isAudio: boolean } {
  switch (extension) {
    case 'mp3':
      return { mimeType: 'audio/mpeg', isAudio: true };
    case 'wav':
      return { mimeType: 'audio/wav', isAudio: true };
    case 'm4a':
      return { mimeType: 'audio/mp4', isAudio: true };
    case 'ogg':
      return { mimeType: 'audio/ogg', isAudio: true };
    case 'webm':
      return { mimeType: 'video/webm', isAudio: false };
    case 'mov':
      return { mimeType: 'video/quicktime', isAudio: false };
    case 'mkv':
      return { mimeType: 'video/x-matroska', isAudio: false };
    case 'avi':
      return { mimeType: 'video/x-msvideo', isAudio: false };
    case 'mp4':
    default:
      return { mimeType: 'video/mp4', isAudio: false };
  }
}

// ============================================================
// BULLETPROOF JSON EXTRACTION
// ============================================================

function extractJSON(raw: string): any {
  if (!raw || !raw.trim()) {
    throw new Error('Empty AI response');
  }

  let text = raw;
  text = text.replace(/^```(?:json)?\s*\n?/im, '').replace(/\n?\s*```\s*$/im, '');

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

  const lastBrace = text.lastIndexOf('}');
  const lastBracket = text.lastIndexOf(']');
  const endIdx = Math.max(lastBrace, lastBracket);
  if (endIdx === -1) {
    throw new Error(`No closing brace/bracket found. First 300 chars: ${text.substring(0, 300)}`);
  }

  text = text.substring(0, endIdx + 1);
  text = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, '');
  text = text.replace(/,\s*([}\]])/g, '$1');

  try {
    return JSON.parse(text);
  } catch {
    console.log('[extractJSON] First parse failed, attempting repair...');
  }

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

  try {
    return JSON.parse(repaired);
  } catch {
    console.error('[extractJSON] Repair failed. First 500 chars:', text.substring(0, 500));
    throw new Error(`Failed to parse AI response as JSON after repair. First 300 chars: ${raw.substring(0, 300)}`);
  }
}

// ============================================================
// SUPABASE HELPERS
// ============================================================

async function supabaseFetch(path: string, options: RequestInit = {}): Promise<Response> {
  return fetch(`${SUPABASE_URL}${path}`, {
    ...options,
    headers: {
      apikey: SUPABASE_SERVICE_KEY,
      Authorization: `Bearer ${SUPABASE_SERVICE_KEY}`,
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });
}

async function pgmqRead(): Promise<QueueMessage | null> {
  const res = await supabaseFetch('/rest/v1/rpc/pgmq_read', {
    method: 'POST',
    body: JSON.stringify({
      queue_name: 'scoring_jobs',
      vt: VISIBILITY_TIMEOUT,
      qty: 1,
    }),
  });

  if (!res.ok) {
    console.error('pgmq_read error:', await res.text());
    return null;
  }

  const rows = (await res.json()) as QueueMessage[];
  return rows?.[0] || null;
}

async function pgmqArchive(msgId: number): Promise<void> {
  const res = await supabaseFetch('/rest/v1/rpc/pgmq_archive', {
    method: 'POST',
    body: JSON.stringify({ queue_name: 'scoring_jobs', msg_id: msgId }),
  });

  if (!res.ok) {
    console.error('pgmq_archive error:', await res.text());
  }
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

async function insertScoreDetails(details: Array<Record<string, any>>): Promise<void> {
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
  const res = await supabaseFetch(`/rest/v1/call_score_details?call_score_id=eq.${scoreId}`, {
    method: 'DELETE',
  });

  if (!res.ok) {
    console.error('Failed to delete existing score details:', await res.text());
  }
}

// ============================================================
// DOWNLOAD RECORDING TO DISK
// ============================================================

function downloadFile(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    const client = url.startsWith('https') ? https : http;

    const request = client.get(url, { timeout: 300_000 }, (response) => {
      if (
        response.statusCode &&
        response.statusCode >= 300 &&
        response.statusCode < 400 &&
        response.headers.location
      ) {
        file.close();
        if (fs.existsSync(dest)) fs.unlinkSync(dest);
        downloadFile(response.headers.location, dest).then(resolve).catch(reject);
        return;
      }

      if (response.statusCode !== 200) {
        file.close();
        if (fs.existsSync(dest)) fs.unlinkSync(dest);
        reject(new Error(`Download failed: ${response.statusCode}`));
        return;
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

    request.on('error', (error) => {
      file.close();
      if (fs.existsSync(dest)) fs.unlinkSync(dest);
      reject(error);
    });

    request.on('timeout', () => {
      request.destroy();
      file.close();
      if (fs.existsSync(dest)) fs.unlinkSync(dest);
      reject(new Error('Download timeout'));
    });
  });
}

// ============================================================
// PROMPT BUILDERS
// ============================================================

function buildMediaAnalysisPrompt(callContext: Record<string, any>, mode: MediaAnalysisMode): string {
  const isAudioOnly = mode === 'audio_only';

  return `Analyze this sales call ${isAudioOnly ? 'audio' : 'recording'} for delivery quality. The rep is ${callContext.rep_name || 'unknown'} and the prospect is ${callContext.contact_name || 'unknown'}.

${isAudioOnly ? 'This upload is audio-only. Do NOT infer camera presence, eye contact, body language, or physical environment. Set visual_analysis to null.' : 'This upload includes video and audio. Use both modalities when available.'}

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
  "visual_analysis": ${isAudioOnly ? 'null' : '{"eye_contact": "", "environment": "", "body_language": ""}'},
  "engagement_timeline": [{"timestamp": "MM:SS", "engagement_level": 1, "note": ""}],
  "active_listening_signals": [""],
  "monologue_flags": [{"timestamp": "MM:SS", "duration_seconds": 0, "speaker": ""}]
}

CRITICAL OUTPUT FORMAT: Return ONLY the raw JSON object. Do NOT include markdown headers, code fences, or any text before or after the JSON.`;
}

function buildScoringSystemPrompt(rubric: Record<string, any>, callContext: Record<string, any>, audioDelivery: any | null): string {
  const steps = Array.isArray(rubric.steps) ? rubric.steps : [];
  const stepsDescription = steps
    .map((step: Record<string, any>) => {
      const criteria = Array.isArray(step.criteria) ? step.criteria.join(', ') : '';
      return `- ${step.key} (${step.label}, max ${step.max_points} pts): ${step.description}\n  Criteria: ${criteria}`;
    })
    .join('\n');

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
      "adherence_percentage": number,
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
// SCORING WITH RETRY
// ============================================================

const MAX_SCORING_ATTEMPTS = 2;

async function scoreTranscriptWithRetry(
  transcript: string,
  rubric: Record<string, any>,
  callContext: Record<string, any>,
  audioDelivery: any | null,
  apiKey: string,
  model: string,
  provider: string,
): Promise<any> {
  const systemPrompt = buildScoringSystemPrompt(rubric, callContext, audioDelivery);
  const baseUserPrompt = `Score this sales call transcript:\n\n${transcript}`;

  for (let attempt = 1; attempt <= MAX_SCORING_ATTEMPTS; attempt += 1) {
    try {
      const promptSuffix =
        attempt > 1
          ? '\n\nCRITICAL: Your previous response was not valid JSON. Return ONLY a raw JSON object. No markdown, no headers, no code fences, no explanatory text. Start with { and end with }.'
          : '';

      const userPrompt = baseUserPrompt + promptSuffix;
      let responseText = '';

      if (provider === 'anthropic') {
        responseText = await callAnthropic(apiKey, model, systemPrompt, userPrompt);
      } else if (provider === 'openai') {
        responseText = await callOpenAI(apiKey, model, systemPrompt, userPrompt);
      } else {
        responseText = await callOpenRouter(apiKey, model, systemPrompt, userPrompt);
      }

      return extractJSON(responseText);
    } catch (error) {
      console.error(
        `Scoring attempt ${attempt}/${MAX_SCORING_ATTEMPTS} failed:`,
        error instanceof Error ? error.message : String(error),
      );

      if (attempt === MAX_SCORING_ATTEMPTS) {
        throw error;
      }

      console.log('Retrying with stronger JSON enforcement...');
    }
  }

  throw new Error('Scoring failed after all attempts');
}

// ============================================================
// LLM PROVIDER CALLS
// ============================================================

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

  if (!res.ok) {
    throw new Error(`Anthropic error: ${res.status} ${await res.text()}`);
  }

  const data = (await res.json()) as AnthropicResponse;
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

  if (!res.ok) {
    throw new Error(`OpenAI error: ${res.status} ${await res.text()}`);
  }

  const data = (await res.json()) as OpenAIResponse;
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

  if (!res.ok) {
    throw new Error(`OpenRouter error: ${res.status} ${await res.text()}`);
  }

  const data = (await res.json()) as OpenRouterResponse;
  return data.choices?.[0]?.message?.content || '';
}

// ============================================================
// REP NOTIFICATION
// ============================================================

async function sendRepNotification(
  callId: string,
  subAccountId: string,
  callScoreId: string,
  callContext: Record<string, any>,
  scorePercentage: number,
  scoringResult: Record<string, any>,
): Promise<void> {
  const repEmail = callContext.rep_email;
  if (!repEmail) return;

  const res = await supabaseFetch(
    `/rest/v1/profiles?email=eq.${encodeURIComponent(repEmail)}&select=id&limit=1`,
  );
  if (!res.ok) return;

  const profiles = (await res.json()) as Array<{ id?: string }>;
  if (!profiles?.[0]?.id) return;

  const grade =
    scorePercentage >= 90 ? 'A' :
    scorePercentage >= 80 ? 'B' :
    scorePercentage >= 70 ? 'C' :
    scorePercentage >= 60 ? 'D' : 'F';

  const severity = scorePercentage >= 80 ? 'success' : scorePercentage >= 50 ? 'warning' : 'error';
  const dealKiller = scoringResult.deal_killer;
  const topFix = Array.isArray(scoringResult.coaching_fixes) ? scoringResult.coaching_fixes[0] : null;
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
}

// ============================================================
// PROCESS A SINGLE SCORING JOB
// ============================================================

async function processJob(job: Record<string, any>): Promise<void> {
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

  const requestedMode = normalizeMediaMode(job.media_analysis_mode);
  let effectiveMode: MediaAnalysisMode = requestedMode;
  let analysisType: AnalysisType = 'transcript_only';
  let audioDelivery: any | null = null;
  let geminiFileName: string | null = null;

  const extension = inferExtensionFromUrl(String(recording_url || 'recording.mp4'));
  const sourceInfo = inferMimeType(extension);
  const inputPath = `/tmp/scoring-${Date.now()}-${Math.random().toString(36).slice(2)}.${extension}`;
  const audioOnlyPath = `/tmp/scoring-audio-${Date.now()}-${Math.random().toString(36).slice(2)}.ogg`;
  const tempFiles = [inputPath, audioOnlyPath];

  console.log(`Processing call ${call_id} | mode: ${requestedMode}`);

  try {
    // ── MEDIA ANALYSIS (skip if transcript_only or no Gemini key) ──
    if (requestedMode !== 'transcript_only' && gemini_api_key && recording_url) {

      // 1. Download recording to disk
      await downloadFile(recording_url, inputPath);

      // 2. Check duration → auto-downgrade if tokens exceed limit
      const durationSeconds = getMediaDurationSeconds(inputPath);
      if (durationSeconds) {
        const estimatedTokens = estimateGeminiTokens(durationSeconds);
        console.log(`Duration: ${Math.round(durationSeconds)}s | Est. tokens: ${estimatedTokens}`);

        if (requestedMode === 'video_and_audio' && exceedsGeminiMediaTokenLimit(durationSeconds)) {
          effectiveMode = 'audio_only';
          console.log(`Auto-downgrade: video_and_audio → audio_only (${estimatedTokens} > 1,000,000 token limit)`);
        }
      }

      // 3. Prepare upload path (extract audio if needed)
      let uploadPath = inputPath;
      let uploadMimeType = sourceInfo.mimeType;

      if (effectiveMode === 'audio_only') {
        if (sourceInfo.isAudio) {
          uploadPath = inputPath;
          uploadMimeType = sourceInfo.mimeType;
        } else {
          console.log('Extracting audio track with ffmpeg...');
          extractAudioToOgg(inputPath, audioOnlyPath);
          uploadPath = audioOnlyPath;
          uploadMimeType = 'audio/ogg';
          console.log(`Audio extracted: ${(fs.statSync(audioOnlyPath).size / 1024 / 1024).toFixed(1)}MB`);
        }
      }

      // 4. Upload to Gemini
      const uploaded = await uploadFileToGemini(uploadPath, gemini_api_key, uploadMimeType);
      geminiFileName = uploaded.name;

      // 5. Wait for ACTIVE
      const isActive = await waitForFileActive(uploaded.name, gemini_api_key);

      if (isActive) {
        // 6. Analyze with Gemini
        const mediaPrompt = buildMediaAnalysisPrompt(call_context || {}, effectiveMode);
        const mediaResult = await analyzeAudioWithGemini(
          uploaded.uri,
          uploadMimeType,
          mediaPrompt,
          gemini_api_key,
          gemini_model || 'gemini-2.5-flash',
        );

        audioDelivery = extractJSON(mediaResult);
        analysisType = mapModeToAnalysisType(effectiveMode);
        console.log(`Gemini media analysis completed (${effectiveMode})`);
      } else {
        console.error('Gemini file never became ACTIVE; falling back to transcript_only');
        effectiveMode = 'transcript_only';
      }

    } else if (requestedMode !== 'transcript_only') {
      console.log('No Gemini key available; falling back to transcript_only');
      effectiveMode = 'transcript_only';
    }

    // ── TRANSCRIPT SCORING ──
    const scoringResult = await scoreTranscriptWithRetry(
      transcript,
      rubric,
      call_context || {},
      audioDelivery,
      scoring_api_key,
      scoring_model,
      scoring_provider,
    );

    // ── WRITE RESULTS ──
    const maxPossible = scoringResult.max_possible_score || rubric.max_total_score;
    const scorePercentage = maxPossible > 0 ? Math.round((scoringResult.overall_score / maxPossible) * 100) : 0;

    const enrichedAudioDelivery = audioDelivery
      ? {
          ...audioDelivery,
          deal_killer: scoringResult.deal_killer || null,
          coaching_fixes: scoringResult.coaching_fixes || [],
          assigned_drill: scoringResult.assigned_drill || null,
        }
      : null;

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

    // ── INSERT STEP DETAILS ──
    if (Array.isArray(scoringResult.step_scores) && scoringResult.step_scores.length > 0) {
      await deleteScoreDetails(call_score_id);
      const details = scoringResult.step_scores.map((step: Record<string, any>) => ({
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

    // ── NOTIFICATIONS (non-blocking) ──
    try {
      await sendRepNotification(call_id, sub_account_id, call_score_id, call_context || {}, scorePercentage, scoringResult);
    } catch (error) {
      console.error('Notification failed (non-blocking):', error);
    }

    try {
      await supabaseFetch('/functions/v1/send-slack-alert', {
        method: 'POST',
        body: JSON.stringify({
          sub_account_id,
          alert_type: 'call_scored',
          data: {
            rep_name: call_context?.rep_name,
            contact_name: call_context?.contact_name,
            overall_score: scoringResult.overall_score,
            max_possible_score: maxPossible,
            score_percentage: scorePercentage,
            framework_name: rubric.name,
            analysis_type: analysisType,
            workspace_id,
            step_scores: (scoringResult.step_scores || []).map((step: Record<string, any>) => ({
              label: step.step_label,
              score: step.score || 0,
              max: step.max_score || 0,
            })),
            delivery_grade: audioDelivery?.delivery_grade || null,
            delivery_summary: audioDelivery?.delivery_summary || null,
            deal_killer: scoringResult.deal_killer || null,
            coaching_fixes: scoringResult.coaching_fixes || [],
            assigned_drill: scoringResult.assigned_drill || null,
          },
        }),
      });
    } catch (error) {
      console.error('Slack alert failed (non-blocking):', error);
    }

    console.log(`Call ${call_id} scored: ${scorePercentage}% (${scoring_provider}/${scoring_model}) via ${analysisType}`);

  } finally {
    // ── CLEANUP ──
    for (const filePath of tempFiles) {
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
    }

    if (geminiFileName && gemini_api_key) {
      await deleteGeminiFile(geminiFileName, gemini_api_key);
    }
  }
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
        } catch (error) {
          console.error(`Job ${msgId} failed (attempt ${readCount}):`, error);

          if (readCount >= MAX_RETRIES) {
            await updateCallScore(job.call_score_id, {
              status: 'failed',
              error_message: `Worker failed after ${MAX_RETRIES} attempts: ${error instanceof Error ? error.message : String(error)}`,
            });
            await pgmqArchive(msgId);
            console.error(`Job ${msgId} permanently failed after ${MAX_RETRIES} attempts`);
          }
        }
      }
    } catch (error) {
      console.error('Poll error:', error);
    }

    await sleep(POLL_INTERVAL_MS);
  }
}

// ============================================================
// HEALTH CHECK + STARTUP
// ============================================================

const port = Number(process.env.PORT || 3000);

const server = http.createServer((req, res) => {
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', uptime: process.uptime() }));
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

server.listen(port, () => {
  console.log(`Health server listening on port ${port}`);
  pollLoop().catch((error) => {
    console.error('Poll loop crashed:', error);
    process.exit(1);
  });
});
