/**
 * Scoring logic ported directly from supabase/functions/score-sales-call/index.ts
 * Includes: callAI, analyzeAudioWithGemini, buildScoringPrompt
 */

import { downloadToTmp, uploadToGemini, waitForActive, deleteGeminiFile, cleanupTmpFile, detectMimeType } from './gemini';

// ============================================================
// AI CALL HELPERS
// ============================================================

export async function callAI(
  prompt: string,
  systemPrompt: string,
  apiKey: string,
  model: string,
  provider: string,
): Promise<string> {
  if (provider === 'anthropic') {
    const resp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
      body: JSON.stringify({ model, max_tokens: 4096, system: systemPrompt, messages: [{ role: 'user', content: prompt }] }),
    });
    if (!resp.ok) throw new Error(`Anthropic error: ${resp.status} ${await resp.text()}`);
    const data: any = await resp.json();
    return data.content[0]?.text || '';
  }

  if (provider === 'openai' || provider === 'openrouter') {
    const baseUrl = provider === 'openrouter' ? 'https://openrouter.ai/api/v1' : 'https://api.openai.com/v1';
    const resp = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
      body: JSON.stringify({
        model, max_tokens: 4096,
        messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: prompt }],
        response_format: { type: 'json_object' },
      }),
    });
    if (!resp.ok) throw new Error(`${provider} error: ${resp.status} ${await resp.text()}`);
    const data: any = await resp.json();
    return data.choices[0]?.message?.content || '';
  }

  if (provider === 'google') {
    const resp = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          systemInstruction: { parts: [{ text: systemPrompt }] },
          contents: [{ role: 'user', parts: [{ text: prompt }] }],
          generationConfig: { maxOutputTokens: 4096, responseMimeType: 'application/json' },
        }),
      },
    );
    if (!resp.ok) throw new Error(`Gemini error: ${resp.status} ${await resp.text()}`);
    const data: any = await resp.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
  }

  throw new Error(`Unsupported provider: ${provider}`);
}

// ============================================================
// AUDIO ANALYSIS VIA GEMINI (streaming from disk)
// ============================================================

export async function analyzeAudioWithGemini(
  recordingUrl: string,
  apiKey: string,
  model: string,
  jobId: string,
): Promise<Record<string, any> | null> {
  const isVideo = /\.(mp4|webm|mov)(\?|$)/i.test(recordingUrl);
  const mimeType = detectMimeType(recordingUrl);

  const systemPrompt = `You are an expert sales call delivery analyst specializing in audio${isVideo ? ' and video' : ''} analysis. Analyze this sales call recording and return a comprehensive JSON object.

You MUST analyze what you actually hear${isVideo ? ' and see' : ''} — do not fabricate or guess.

Return this exact JSON structure:
{
  "tone": "warm/neutral/aggressive/monotone/enthusiastic",
  "pacing": "too_fast/good/too_slow/varied",
  "talk_to_listen_ratio": 0.65,
  "energy_level": "high/medium/low",
  "monologue_flags": ["2:30-4:15 rep spoke for 1m45s without pause"],
  "delivery_notes": "Brief overall delivery assessment",
  "filler_word_count": 12,
  "filler_examples": ["um at 1:23", "uh at 3:45"],
  "vocal_confidence": "high/medium/low",
  "voice_variation": "dynamic/moderate/flat",
  "enthusiasm_level": "high/medium/low",
  "rapport_quality": "strong/moderate/weak/none",
  "emotional_shifts": [
    { "timestamp": "2:30", "shift": "warm→defensive", "trigger": "price objection raised" }
  ],
  "interruption_count": { "rep": 2, "prospect": 1 },
  "strategic_pauses": 3,
  "awkward_silences": [
    { "timestamp": "5:12", "duration_seconds": 8 }
  ],
  "active_listening_signals": ["mmhm at 3:20"],
  ${isVideo ? `"visual_analysis": {
    "eye_contact_quality": "strong/moderate/poor",
    "body_language": "open/neutral/closed",
    "professionalism": "high/medium/low",
    "environment_notes": "description",
    "notable_gestures": []
  },` : `"visual_analysis": null,`}
  "engagement_timeline": [
    { "minute": 1, "rep_energy": "high/medium/low", "prospect_engagement": "high/medium/low/disengaged", "notes": "brief note" }
  ],
  "critical_moments": [
    { "timestamp": "4:30", "type": "objection_raised/rapport_peak/energy_drop/closing_attempt/breakthrough", "description": "What happened", "rep_response_quality": "excellent/good/fair/poor" }
  ],
  "delivery_grade": "A/B/C/D/F",
  "delivery_summary": "2-3 sentence summary",
  "verification": {
    "speaker_count": 2,
    "first_words_spoken": "the first 8-10 words",
    "approx_duration_seconds": 120,
    "speaker_names_heard": []
  }
}

IMPORTANT: The "verification" block is critical — report what you actually hear. Only return JSON.`;

  let tmpFile: string | null = null;
  let uploadedFileName: string | null = null;

  try {
    // Step 1: Stream download to disk
    const { filePath } = await downloadToTmp(recordingUrl, jobId);
    tmpFile = filePath;

    // Step 2: Chunked upload to Gemini
    const { fileUri, fileName } = await uploadToGemini(filePath, apiKey, mimeType);
    uploadedFileName = fileName;

    // Step 3: Wait for processing
    const isActive = await waitForActive(fileName, apiKey);
    if (!isActive) {
      console.error('Gemini file never became ACTIVE');
      return null;
    }

    // Step 4: Analyze
    const resp = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          systemInstruction: { parts: [{ text: systemPrompt }] },
          contents: [{
            role: 'user',
            parts: [
              { text: 'Analyze the delivery of this sales call:' },
              { fileData: { mimeType, fileUri } },
            ],
          }],
          generationConfig: { maxOutputTokens: 8192, responseMimeType: 'application/json' },
        }),
      },
    );

    if (!resp.ok) {
      const errBody = await resp.text();
      console.error('Gemini generateContent error:', resp.status, errBody);
      return null;
    }

    const data: any = await resp.json();
    const result = JSON.parse(data.candidates?.[0]?.content?.parts?.[0]?.text || '{}');
    console.log('GEMINI VERIFICATION:', JSON.stringify(result.verification || null));
    const { verification: _v, ...cleanResult } = result;
    return cleanResult;
  } catch (err) {
    console.error('Audio analysis failed:', err);
    return null;
  } finally {
    if (tmpFile) cleanupTmpFile(tmpFile);
    if (uploadedFileName) deleteGeminiFile(uploadedFileName, apiKey);
  }
}

// ============================================================
// SCORING PROMPT (ported from edge function)
// ============================================================

export function buildScoringPrompt(
  transcript: string,
  rubric: { steps: any[]; script_content?: string; script_sections?: any[] },
  callContext: { rep_name?: string; contact_name?: string; outcome_type?: string },
  audioInsights?: Record<string, any> | null,
): { system: string; user: string } {
  const maxChars = 400000;
  const truncated = transcript.length > maxChars
    ? transcript.slice(0, maxChars) + '\n\n[Transcript truncated due to length]'
    : transcript;

  let system = `You are an expert sales call scoring analyst. Score the following call transcript against the provided rubric framework.

Return a JSON object with exactly this structure:
{
  "step_scores": [
    {
      "step_key": "string",
      "step_label": "string",
      "score": number,
      "max_score": number,
      "reasoning": "string (2-3 sentences with [mm:ss] timestamps — if audio data available, mention how delivery affected this step)",
      "evidence": ["exact quote with [mm:ss] timestamp"],
      "adherence_percentage": number or null,
      "key_phrases_hit": ["phrase1"] or null,
      "key_phrases_missed": ["phrase1"] or null
    }
  ],
  "overall_score": number,
  "max_possible_score": number,
  "objections_detected": [
    {
      "objection": "string",
      "handled_well": boolean,
      "response_quality": "excellent/good/fair/poor",
      "evidence": "exact quote with [mm:ss] timestamp"
    }
  ],
  "deal_killer": {
    "summary": "string (the single biggest reason this deal was lost or weakened)",
    "timestamp": "[mm:ss]",
    "quote": "exact words",
    "what_to_do_instead": "specific alternative with example language"
  },
  "coaching_fixes": [
    {
      "issue": "string (what went wrong with [mm:ss] timestamp and quote)",
      "fix": "string (exact alternative behavior with example language)",
      "script_reference": "string or null"
    }
  ],
  "assigned_drill": {
    "name": "string (memorable drill name)",
    "why": "string (which metric was weakest and why this drill fixes it)"
  },
  "coaching_notes": "string (overall narrative coaching summary)",
  "strengths": ["string"],
  "improvements": ["string"]
}

CRITICAL SCORING RULES:
1. TIMESTAMPS REQUIRED: Every piece of evidence MUST include [mm:ss] timestamps.
2. SILENCE DISCIPLINE: If talk_to_listen_ratio > 0.45, penalize. If strategic_pauses = 0 after close attempts, flag it.
3. PRE-DISQUALIFICATION DETECTION: If the rep objects FOR the prospect, flag in deal_killer or coaching_fixes.
4. DEAL KILLER: Always identify the single biggest moment that cost the deal.
5. COACHING FIXES: Provide 2-4 specific fixes with timestamps and example language.
6. ASSIGNED DRILL: Based on the weakest metric, assign a specific practice drill.

Score fairly and specifically. Use exact transcript quotes. Be direct.`;

  if (rubric.script_content) {
    system += `\n\nSales script to measure adherence:\n--- SALES SCRIPT ---\n${rubric.script_content}\n--- END SCRIPT ---`;
  }

  if (rubric.script_sections && rubric.script_sections.length > 0) {
    system += `\n\nScript section-to-rubric mapping:\n${JSON.stringify(rubric.script_sections, null, 2)}\n\nFor each mapped section, calculate adherence_percentage and track key_phrases_hit/key_phrases_missed.`;
  }

  const rubricSteps = rubric.steps.map((s: any) =>
    `- ${s.key} (${s.label}): ${s.description} [max ${s.max_points} pts]\n  Criteria: ${(s.criteria || []).join(', ')}`
  ).join('\n');

  let user = `## Call Context
Rep: ${callContext.rep_name || 'Unknown'}
Contact: ${callContext.contact_name || 'Unknown'}
Outcome: ${callContext.outcome_type || 'Unknown'}

## Rubric Framework
${rubricSteps}`;

  if (audioInsights) {
    system += `\n\nIMPORTANT: Audio/video delivery analysis has been performed. Factor these insights into scoring:
- Poor delivery (low confidence, flat energy) should score LOWER even with right words.
- Strong delivery can elevate scores.
- If talk_to_listen_ratio > 0.45, rep talks too much — penalize discovery/rapport.
- Watch for pre-disqualification behaviors.`;

    user += `\n\n## Audio/Video Delivery Analysis
Delivery Grade: ${audioInsights.delivery_grade || 'N/A'}
Vocal Confidence: ${audioInsights.vocal_confidence || 'N/A'}
Tone: ${audioInsights.tone || 'N/A'}
Energy: ${audioInsights.energy_level || 'N/A'}
Rapport: ${audioInsights.rapport_quality || 'N/A'}
Filler Words: ${audioInsights.filler_word_count ?? 'N/A'}
Enthusiasm: ${audioInsights.enthusiasm_level || 'N/A'}
Interruptions: ${audioInsights.interruption_count ? `rep=${audioInsights.interruption_count.rep}/prospect=${audioInsights.interruption_count.prospect}` : 'N/A'}
Strategic Pauses: ${audioInsights.strategic_pauses ?? 'N/A'}
Summary: ${audioInsights.delivery_summary || audioInsights.delivery_notes || 'N/A'}`;

    if (audioInsights.critical_moments?.length) {
      user += `\n\nCritical Moments:\n${audioInsights.critical_moments.map((m: any) => `- [${m.timestamp}] ${m.type}: ${m.description} (${m.rep_response_quality})`).join('\n')}`;
    }
    if (audioInsights.emotional_shifts?.length) {
      user += `\n\nEmotional Shifts:\n${audioInsights.emotional_shifts.map((s: any) => `- [${s.timestamp}] ${s.shift} — ${s.trigger}`).join('\n')}`;
    }
    if (audioInsights.visual_analysis) {
      const v = audioInsights.visual_analysis;
      user += `\n\nVisual: Eye contact=${v.eye_contact_quality}, Body=${v.body_language}, Professional=${v.professionalism}`;
    }
  }

  user += `\n\n## Transcript\n${truncated}`;
  return { system, user };
}
