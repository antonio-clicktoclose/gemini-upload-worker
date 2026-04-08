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
const VISIBILITY_TIMEOUT = 720;

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
// COACHING DATA NORMALIZER
// ============================================================

function splitPlainStringToIssueFix(text: string): { issue: string; fix: string; script_reference: string | null } {
  const separators = ['Instead:', 'Fix:', '->', '→', ' — '];
  for (const sep of separators) {
    const idx = text.indexOf(sep);
    if (idx > 0) {
      return {
        issue: text.substring(0, idx).trim(),
        fix: text.substring(idx + sep.length).trim(),
        script_reference: null,
      };
    }
  }
  return { issue: text, fix: '', script_reference: null };
}

function normalizeCoachingData(scoringResult: any): void {
  if (scoringResult.coaching_fixes && Array.isArray(scoringResult.coaching_fixes)) {
    scoringResult.coaching_fixes = scoringResult.coaching_fixes.map((fix: any) => {
      if (typeof fix === 'string') return splitPlainStringToIssueFix(fix);
      return fix;
    });
  }
  if (typeof scoringResult.deal_killer === 'string') {
    scoringResult.deal_killer = { summary: scoringResult.deal_killer, timestamp: null, quote: null, what_to_do_instead: null };
  }
  if (typeof scoringResult.assigned_drill === 'string') {
    scoringResult.assigned_drill = { name: scoringResult.assigned_drill, why: null };
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
  "delivery_grade": "A/B/C/D/F",
  "delivery_summary": "2-3 sentence summary of overall delivery quality including vocal presence, emotional intelligence, and conversation control",
  "tone": "warm/neutral/aggressive/monotone/enthusiastic",
  "energy_level": "high/medium/low",
  "vocal_confidence": "high/medium/low",
  "pacing": "too_fast/good/too_slow/varied",
  "enthusiasm_level": "high/medium/low",
  "rapport_quality": "strong/moderate/weak/none",
  "voice_variation": "dynamic/moderate/flat",
  "talk_to_listen_ratio": 0.65,
  "filler_word_count": 12,
  "filler_examples": ["um at 1:23", "uh at 3:45", "like at 5:02"],
  "strategic_pauses": 3,
  "interruption_count": { "rep": 2, "prospect": 1 },
  "emotional_shifts": [{"timestamp": "MM:SS", "shift": "warm→defensive", "trigger": "price objection raised"}],
  "critical_moments": [{"timestamp": "MM:SS", "type": "objection_raised/rapport_peak/energy_drop/closing_attempt/breakthrough", "description": "What happened", "rep_response_quality": "excellent/good/fair/poor"}],
  "visual_analysis": ${isAudioOnly ? 'null' : '{"eye_contact_quality": "strong/moderate/poor", "body_language": "open/neutral/closed", "professionalism": "high/medium/low", "environment_notes": "description", "notable_gestures": ["nodded at 3:45"]}'},
  "engagement_timeline": [{"minute": 1, "rep_energy": "high/medium/low", "prospect_engagement": "high/medium/low/disengaged", "notes": "brief note"}],
  "active_listening_signals": ["mmhm at 3:20", "repeated prospect's words at 4:10"],
  "monologue_flags": ["2:30-4:15 rep spoke for 1m45s without pause"],
  "delivery_notes": "Brief overall delivery assessment"
}

IMPORTANT RULES:
- engagement_timeline should have one entry per minute of the call (up to 60 entries max).
- critical_moments should capture 3-8 key turning points.
- filler_examples: list up to 10 most notable instances with timestamps.
- emotional_shifts: only include genuine shifts you detect, not every moment.
- Only return the JSON object, no other text.
- FORMAT ENFORCEMENT: For tone/pacing/energy_level/vocal_confidence/rapport_quality/voice_variation/enthusiasm_level, return ONLY the single-word or short label from the options shown in the schema (e.g. "warm", "good", "high") — NEVER return a full sentence or paragraph.
- talk_to_listen_ratio MUST be a decimal number (e.g. 0.65), NOT a string like "65:35".
- interruption_count MUST be an object {"rep": N, "prospect": N}, NEVER a plain number.
- filler_word_count and strategic_pauses MUST be plain integers, not strings.

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

  let audioContext = '';
  if (audioDelivery) {
    audioContext = `\n\nAudio/Video Delivery Analysis Available:
Delivery Grade: ${audioDelivery.delivery_grade || 'N/A'}
Vocal Confidence: ${audioDelivery.vocal_confidence || 'N/A'}
Tone: ${audioDelivery.tone || 'N/A'}
Energy: ${audioDelivery.energy_level || 'N/A'}
Rapport Quality: ${audioDelivery.rapport_quality || 'N/A'}
Filler Words: ${audioDelivery.filler_word_count ?? 'N/A'}
Enthusiasm: ${audioDelivery.enthusiasm_level || 'N/A'}
Interruptions (rep/prospect): ${audioDelivery.interruption_count ? `${audioDelivery.interruption_count.rep}/${audioDelivery.interruption_count.prospect}` : 'N/A'}
Strategic Pauses: ${audioDelivery.strategic_pauses ?? 'N/A'}
Talk-to-Listen Ratio: ${audioDelivery.talk_to_listen_ratio ?? 'N/A'}
Delivery Summary: ${audioDelivery.delivery_summary || audioDelivery.delivery_notes || 'N/A'}

IMPORTANT: Factor these delivery insights into your scoring:
- A rep who says the right words but delivers them poorly (low confidence, flat energy, no rapport) should score LOWER.
- Strong delivery can elevate scores for steps where the rep showed genuine engagement and authority.
- SILENCE DISCIPLINE: Check strategic_pauses count. If strategic_pauses = 0 after close attempts or price reveals, this is a critical failure.
- TALK RATIO: If talk_to_listen_ratio > 0.45, the rep is talking too much. Flag and penalize discovery/rapport steps.
- PRE-DISQUALIFICATION: Watch for moments where the rep objects on behalf of the prospect. This is a deal-killing behavior.`;

    if (audioDelivery.critical_moments?.length) {
      audioContext += `\n\nCritical Moments:\n${audioDelivery.critical_moments.map((m: any) => `- [${m.timestamp}] ${m.type || m.impact}: ${m.description} (rep response: ${m.rep_response_quality || 'N/A'})`).join('\n')}`;
    }

    if (audioDelivery.emotional_shifts?.length) {
      audioContext += `\n\nEmotional Shifts:\n${audioDelivery.emotional_shifts.map((s: any) => `- [${s.timestamp}] ${s.shift || `${s.from_emotion}→${s.to_emotion}`} — trigger: ${s.trigger}`).join('\n')}`;
    }

    if (audioDelivery.visual_analysis) {
      const v = audioDelivery.visual_analysis;
      audioContext += `\n\nVisual Analysis: Eye contact: ${v.eye_contact_quality || v.eye_contact}, Body language: ${v.body_language}, Professionalism: ${v.professionalism || 'N/A'}`;
      if (v.environment_notes || v.environment) audioContext += `, Environment: ${v.environment_notes || v.environment}`;
    }
  }

  return `You are an expert sales call scoring analyst. Score this call using the "${rubric.name}" framework.

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
      "reasoning": "string (2-3 sentences explaining the score with [mm:ss] timestamp references — if audio/video data is available, mention how delivery quality affected this step's score)",
      "evidence": ["exact quote with [mm:ss] timestamp"],
      "adherence_percentage": number,
      "key_phrases_hit": ["string"],
      "key_phrases_missed": ["string"]
    }
  ],
  "objections_detected": [
    {
      "objection": "string",
      "handled_well": boolean,
      "response_quality": "excellent/good/fair/poor",
      "evidence": "exact quote with [mm:ss] timestamp"
    }
  ],
  "strengths": ["string"],
  "improvements": ["string"],
  "coaching_notes": "string (overall narrative coaching summary — be direct and specific about what happened and why the score is what it is)",
  "deal_killer": {
    "summary": "string (the single biggest reason this deal was lost or weakened)",
    "timestamp": "[mm:ss]",
    "quote": "exact words the rep said at that moment",
    "what_to_do_instead": "string (specific alternative behavior with example language)"
  },
  "coaching_fixes": [
    {
      "issue": "string (what went wrong — include [mm:ss] timestamp and exact quote)",
      "fix": "string (exact alternative behavior to practice, with example language)",
      "script_reference": "string or null (which script/rubric section this maps to)"
    }
  ],
  "assigned_drill": {
    "name": "string (a memorable drill name like 'The 8-Second Torture', 'The Mirror Close', 'The Silence Challenge')",
    "why": "string (which metric was weakest and exactly why this drill fixes it — reference specific data)"
  }
}

CRITICAL SCORING RULES:
1. TIMESTAMPS REQUIRED: Every piece of evidence, every quote, every reasoning reference MUST include [mm:ss] timestamps.
2. SILENCE DISCIPLINE: If audio data shows talk_to_listen_ratio > 0.45 (rep talking >45%), penalize accordingly. If strategic_pauses = 0, flag it.
3. PRE-DISQUALIFICATION DETECTION: If the rep objects FOR the prospect, flag it in deal_killer or coaching_fixes.
4. DEAL KILLER: Always identify the single biggest moment that cost the deal or weakened the outcome. Be brutally honest.
5. COACHING FIXES: Provide 2-4 specific, actionable fixes with timestamps. Each fix must have a concrete "do this instead" with example language.
6. ASSIGNED DRILL: Based on the weakest metric, assign a specific practice drill.
7. STRICT FORMAT: Each "coaching_fixes" entry MUST be a JSON object with "issue", "fix", and "script_reference" keys. NEVER return plain strings in the coaching_fixes array.
8. "deal_killer" MUST be a JSON object with "summary", "timestamp", "quote", "what_to_do_instead". NEVER a plain string.
9. "assigned_drill" MUST be a JSON object with "name" and "why". NEVER a plain string.

Score fairly and specifically. Use exact transcript quotes as evidence. Be direct — not generic.

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

      const result = extractJSON(responseText);
      // Normalize coaching data — ensure structured objects even if model returned plain strings
      normalizeCoachingData(result);
      return result;
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
    scorePercentage >= 80 ? 'A' :
    scorePercentage >= 65 ? 'B' :
    scorePercentage >= 50 ? 'C' :
    scorePercentage >= 35 ? 'D' : 'F';

  const severity = scorePercentage >= 80 ? 'success' : scorePercentage >= 50 ? 'warning' : 'error';
  const dealKiller = scoringResult.deal_killer;
  const topFix = Array.isArray(scoringResult.coaching_fixes) ? scoringResult.coaching_fixes[0] : null;
  const takeaway = dealKiller
    ? `Deal killer: ${typeof dealKiller === 'string' ? dealKiller : dealKiller.summary || JSON.stringify(dealKiller)}`
    : topFix
    ? `Top fix: ${typeof topFix === 'string' ? topFix : topFix.issue || JSON.stringify(topFix)}`
    : '';
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
