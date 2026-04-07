/**
 * Scoring Engine — processes a single PGMQ job
 * 
 * Handles: disk-streaming downloads, ffmpeg audio extraction,
 * Gemini File API upload (8MB chunks), token estimation,
 * media mode branching, and LLM transcript scoring.
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';

// ── Constants ──
const CHUNK_SIZE = 8 * 1024 * 1024; // 8MB upload chunks
const ACTIVE_WAIT_TIMEOUT_MS = 300_000; // 5 min for Gemini to process
const GENERATE_CONTENT_TIMEOUT_MS = 600_000; // 10 min for LLM response
const TOKEN_LIMIT = 1_000_000; // Gemini 2.5 Flash context limit (safe threshold)

// ============================================================
// MAIN JOB PROCESSOR
// ============================================================

export async function processJob(supabase: any, job: any): Promise<void> {
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
    file_size,
    media_analysis_mode,
  } = job;

  // Resolve effective media mode
  let effectiveMode: string = media_analysis_mode || 'video_and_audio';

  // Update status to processing
  await supabase.from('call_scores').update({
    status: 'processing',
    updated_at: new Date().toISOString(),
  }).eq('id', call_score_id);

  let audioDelivery: Record<string, any> | null = null;
  let analysisType = 'transcript_only';

  // ── MEDIA ANALYSIS ──
  if (recording_url && gemini_api_key && effectiveMode !== 'transcript_only') {

    // Token estimation: auto-downgrade video_and_audio if too long
    if (effectiveMode === 'video_and_audio' && file_size) {
      // ~100KB/sec for compressed video → estimate duration
      const estimatedDurationSec = Math.ceil(file_size / 100_000);
      const estimatedTokens = estimatedDurationSec * 290;
      if (estimatedTokens > TOKEN_LIMIT) {
        console.log(
          `Token estimate ${estimatedTokens.toLocaleString()} > ${TOKEN_LIMIT.toLocaleString()} ` +
          `(~${Math.round(estimatedDurationSec / 60)}min video) — downgrading to audio_only`
        );
        effectiveMode = 'audio_only';
      }
    }

    // Download recording to disk
    const inputPath = path.join('/tmp', `input_${call_id}`);
    let uploadPath = inputPath;
    let uploadMimeType = detectMimeType(recording_url);

    try {
      console.log(`Downloading recording to ${inputPath}...`);
      await downloadToFile(recording_url, inputPath);
      const actualSize = fs.statSync(inputPath).size;
      console.log(`Downloaded: ${(actualSize / 1024 / 1024).toFixed(1)}MB`);

      // ── FFMPEG AUDIO EXTRACTION for audio_only mode ──
      if (effectiveMode === 'audio_only') {
        const audioPath = path.join('/tmp', `audio_${call_id}.ogg`);
        console.log('Extracting audio with ffmpeg...');
        try {
          execSync(
            `ffmpeg -y -i "${inputPath}" -vn -acodec libopus -b:a 48k "${audioPath}"`,
            { timeout: 120_000, stdio: 'pipe' }
          );
          const audioSize = fs.statSync(audioPath).size;
          console.log(`Audio extracted: ${(audioSize / 1024 / 1024).toFixed(1)}MB (from ${(actualSize / 1024 / 1024).toFixed(1)}MB video)`);
          uploadPath = audioPath;
          uploadMimeType = 'audio/ogg';
        } catch (ffmpegErr) {
          console.error('ffmpeg extraction failed, falling back to full file upload:', ffmpegErr);
          // Continue with video file — Gemini can still analyze audio from it
        }
      }

      // ── UPLOAD TO GEMINI FILE API ──
      const uploaded = await uploadToGeminiChunked(uploadPath, gemini_api_key, uploadMimeType);
      if (!uploaded) {
        console.error('Gemini upload failed — falling back to transcript_only');
      } else {
        // Wait for ACTIVE
        const isActive = await waitForFileActive(uploaded.fileName, gemini_api_key);
        if (isActive) {
          // Analyze
          audioDelivery = await analyzeWithGemini(
            uploaded.fileUri,
            uploadMimeType,
            gemini_api_key,
            gemini_model || 'gemini-2.5-flash',
            effectiveMode === 'video_and_audio', // includeVisual
          );
          if (audioDelivery) {
            analysisType = effectiveMode === 'audio_only' ? 'audio_only' : 'audio_and_transcript';
          }
        }
        // Cleanup Gemini file
        deleteGeminiFile(uploaded.fileName, gemini_api_key);
      }
    } catch (mediaErr) {
      console.error('Media analysis failed (continuing with transcript):', mediaErr);
    } finally {
      // Cleanup local files
      try { if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath); } catch (_) {}
      const audioPath = path.join('/tmp', `audio_${call_id}.ogg`);
      try { if (fs.existsSync(audioPath)) fs.unlinkSync(audioPath); } catch (_) {}
    }
  } else if (!gemini_api_key && effectiveMode !== 'transcript_only') {
    console.log('No Gemini API key available — falling back to transcript_only');
    effectiveMode = 'transcript_only';
  }

  // ── TRANSCRIPT SCORING ──
  const { system, user } = buildScoringPrompt(transcript, rubric, call_context, audioDelivery);

  let scoringResult: any;
  const MAX_ATTEMPTS = 2;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      const promptSuffix = attempt > 1
        ? '\n\nCRITICAL: Your previous response was not valid JSON. Return ONLY a raw JSON object. No markdown, no headers, no code fences. Start with { and end with }.'
        : '';
      const rawResult = await callAI(user + promptSuffix, system, scoring_api_key, scoring_model, scoring_provider);
      scoringResult = extractJSON(rawResult);
      break;
    } catch (e) {
      console.error(`Scoring attempt ${attempt}/${MAX_ATTEMPTS} failed:`, e instanceof Error ? e.message : String(e));
      if (attempt === MAX_ATTEMPTS) {
        await supabase.from('call_scores').update({
          status: 'failed',
          error_message: `AI scoring failed: ${e instanceof Error ? e.message : String(e)}`,
          ai_provider: scoring_provider,
          ai_model: scoring_model,
          updated_at: new Date().toISOString(),
        }).eq('id', call_score_id);
        throw e;
      }
    }
  }

  // ── WRITE RESULTS ──
  const overallScore = scoringResult.overall_score || 0;
  const maxPossible = scoringResult.max_possible_score || rubric.max_total_score;

  const enrichedAudioDelivery = {
    ...(audioDelivery || {}),
    deal_killer: scoringResult.deal_killer || null,
    coaching_fixes: scoringResult.coaching_fixes || [],
    assigned_drill: scoringResult.assigned_drill || null,
  };

  const { error: updateError } = await supabase.from('call_scores').update({
    rubric_id: rubric.id,
    overall_score: overallScore,
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
    updated_at: new Date().toISOString(),
    error_message: null,
  }).eq('id', call_score_id);

  if (updateError) throw new Error(`Failed to save score: ${updateError.message}`);

  // Insert step details
  if (scoringResult.step_scores && Array.isArray(scoringResult.step_scores)) {
    await supabase.from('call_score_details').delete().eq('call_score_id', call_score_id);

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

    const { error: detailError } = await supabase.from('call_score_details').insert(details);
    if (detailError) console.error('Failed to insert score details:', detailError);
  }

  // Trigger Slack alert
  try {
    const supabaseUrl = process.env.SUPABASE_URL!;
    const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
    const scorePercentage = maxPossible > 0 ? Math.round((overallScore / maxPossible) * 100) : 0;

    await fetch(`${supabaseUrl}/functions/v1/send-slack-alert`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${serviceKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sub_account_id,
        alert_type: 'call_scored',
        data: {
          rep_name: call_context.rep_name,
          contact_name: call_context.contact_name,
          overall_score: overallScore,
          max_possible_score: maxPossible,
          score_percentage: scorePercentage,
          framework_name: rubric.name,
          analysis_type: analysisType,
          ai_model: `${scoring_provider}/${scoring_model}`,
          step_scores: (scoringResult.step_scores || []).map((s: any) => ({
            label: s.step_label,
            score: s.score || 0,
            max: s.max_score || 0,
          })),
          delivery_grade: audioDelivery?.delivery_grade || null,
          delivery_summary: audioDelivery?.delivery_summary || null,
          tone: audioDelivery?.tone || null,
          energy_level: audioDelivery?.energy_level || null,
          vocal_confidence: audioDelivery?.vocal_confidence || null,
          pacing: audioDelivery?.pacing || null,
          enthusiasm_level: audioDelivery?.enthusiasm_level || null,
          rapport_quality: audioDelivery?.rapport_quality || null,
          voice_variation: audioDelivery?.voice_variation || null,
          talk_to_listen_ratio: audioDelivery?.talk_to_listen_ratio || null,
          filler_word_count: audioDelivery?.filler_word_count ?? null,
          strategic_pauses: audioDelivery?.strategic_pauses ?? null,
          interruption_count: audioDelivery?.interruption_count || null,
          emotional_shifts: audioDelivery?.emotional_shifts || [],
          critical_moments: audioDelivery?.critical_moments || [],
          visual_analysis: audioDelivery?.visual_analysis || null,
          engagement_timeline: audioDelivery?.engagement_timeline || [],
          active_listening_signals: audioDelivery?.active_listening_signals || [],
          monologue_flags: audioDelivery?.monologue_flags || [],
          objections: (scoringResult.objections_detected || []).map((o: any) => ({
            objection: o.objection,
            handled: o.handled_well,
            quality: o.response_quality,
          })),
          strengths: scoringResult.strengths || [],
          improvements: scoringResult.improvements || [],
          coaching_notes: scoringResult.coaching_notes,
          deal_killer: scoringResult.deal_killer || null,
          coaching_fixes: scoringResult.coaching_fixes || [],
          assigned_drill: scoringResult.assigned_drill || null,
        },
      }),
    });
  } catch (slackErr) {
    console.error('Slack alert failed (non-blocking):', slackErr);
  }

  // In-app notification
  try {
    const scorePercentage = maxPossible > 0 ? Math.round((overallScore / maxPossible) * 100) : 0;
    const grade = scorePercentage >= 90 ? 'A' : scorePercentage >= 80 ? 'B' : scorePercentage >= 70 ? 'C' : scorePercentage >= 60 ? 'D' : 'F';
    const severity = scorePercentage >= 80 ? 'success' : scorePercentage >= 50 ? 'warning' : 'error';
    const dealKiller = scoringResult.deal_killer;
    const topFix = (scoringResult.coaching_fixes || [])[0];
    const takeaway = dealKiller ? `Deal killer: ${dealKiller}` : topFix ? `Top fix: ${topFix}` : '';
    const messageBody = takeaway ? `${takeaway} | call_id:${call_id}` : `call_id:${call_id}`;

    // Look up rep by email from call_context
    if (call_context.rep_email) {
      const { data: repProfile } = await supabase
        .from('profiles')
        .select('id')
        .eq('email', call_context.rep_email)
        .maybeSingle();

      if (repProfile?.id) {
        await supabase.from('ai_insight_notifications').insert({
          user_id: repProfile.id,
          sub_account_id,
          insight_type: 'call_scored',
          insight_hash: `call_scored_${call_score_id}`,
          title: `Call Scored: ${call_context.contact_name || 'Unknown'} — ${scorePercentage}% (${grade})`,
          message: messageBody,
          severity,
          notification_channel: 'in_app',
          sent_at: new Date().toISOString(),
        });
      }
    }
  } catch (notifErr) {
    console.error('Notification failed (non-blocking):', notifErr);
  }

  const scorePercentage = maxPossible > 0 ? Math.round((overallScore / maxPossible) * 100) : 0;
  console.log(`Call ${call_id} scored: ${scorePercentage}% (${scoring_provider}/${scoring_model}) mode=${analysisType}`);
}

// ============================================================
// HELPERS
// ============================================================

function detectMimeType(url: string): string {
  const ext = url.split('?')[0].split('.').pop()?.toLowerCase() || '';
  const mimeMap: Record<string, string> = {
    mp4: 'video/mp4', webm: 'video/webm', mov: 'video/quicktime',
    mp3: 'audio/mpeg', wav: 'audio/wav', m4a: 'audio/mp4',
    ogg: 'audio/ogg', flac: 'audio/flac',
  };
  return mimeMap[ext] || 'video/mp4';
}

async function downloadToFile(url: string, filePath: string): Promise<void> {
  const resp = await fetch(url);
  if (!resp.ok || !resp.body) throw new Error(`Download failed: ${resp.status}`);
  const fileStream = fs.createWriteStream(filePath);
  // @ts-ignore - Node 20 supports Readable.fromWeb
  await pipeline(Readable.fromWeb(resp.body as any), fileStream);
}

// ============================================================
// GEMINI FILE API — CHUNKED UPLOAD
// ============================================================

async function uploadToGeminiChunked(
  filePath: string,
  apiKey: string,
  mimeType: string,
): Promise<{ fileUri: string; fileName: string } | null> {
  try {
    const fileSize = fs.statSync(filePath).size;
    console.log(`Uploading ${(fileSize / 1024 / 1024).toFixed(1)}MB to Gemini File API (${mimeType})...`);

    // For files small enough, use simple upload
    if (fileSize < CHUNK_SIZE * 2) {
      const body = fs.readFileSync(filePath);
      const resp = await fetch(
        `https://generativelanguage.googleapis.com/upload/v1beta/files?key=${apiKey}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': mimeType,
            'X-Goog-Upload-Protocol': 'raw',
            'X-Goog-Upload-Command': 'upload, finalize',
            'Content-Length': fileSize.toString(),
          },
          body,
        },
      );
      if (!resp.ok) {
        console.error('Gemini simple upload failed:', resp.status, await resp.text());
        return null;
      }
      const result = await resp.json();
      const file = result.file;
      if (!file?.uri || !file?.name) return null;
      console.log(`Uploaded: ${file.name} (simple)`);
      return { fileUri: file.uri, fileName: file.name };
    }

    // Resumable upload for large files
    // Step 1: Initiate
    const initResp = await fetch(
      `https://generativelanguage.googleapis.com/upload/v1beta/files?key=${apiKey}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Goog-Upload-Protocol': 'resumable',
          'X-Goog-Upload-Command': 'start',
          'X-Goog-Upload-Header-Content-Length': fileSize.toString(),
          'X-Goog-Upload-Header-Content-Type': mimeType,
        },
        body: JSON.stringify({ file: { display_name: path.basename(filePath) } }),
      },
    );

    if (!initResp.ok) {
      console.error('Gemini resumable init failed:', initResp.status, await initResp.text());
      return null;
    }

    const uploadUrl = initResp.headers.get('x-goog-upload-url');
    if (!uploadUrl) {
      console.error('No upload URL in response headers');
      return null;
    }
    await initResp.text(); // consume body

    // Step 2: Upload chunks
    const fd = fs.openSync(filePath, 'r');
    let offset = 0;

    while (offset < fileSize) {
      const remaining = fileSize - offset;
      const chunkSize = Math.min(CHUNK_SIZE, remaining);
      const isLast = offset + chunkSize >= fileSize;
      const buffer = Buffer.alloc(chunkSize);
      fs.readSync(fd, buffer, 0, chunkSize, offset);

      const command = isLast ? 'upload, finalize' : 'upload';
      const chunkResp = await fetch(uploadUrl, {
        method: 'POST',
        headers: {
          'Content-Length': chunkSize.toString(),
          'X-Goog-Upload-Offset': offset.toString(),
          'X-Goog-Upload-Command': command,
        },
        body: buffer,
      });

      if (!chunkResp.ok && !isLast) {
        console.error(`Chunk upload failed at offset ${offset}:`, chunkResp.status);
        fs.closeSync(fd);
        return null;
      }

      if (isLast) {
        const result = await chunkResp.json();
        fs.closeSync(fd);
        const file = result.file;
        if (!file?.uri || !file?.name) return null;
        console.log(`Uploaded: ${file.name} (chunked, ${Math.ceil(fileSize / CHUNK_SIZE)} chunks)`);
        return { fileUri: file.uri, fileName: file.name };
      } else {
        await chunkResp.text(); // consume body
      }

      offset += chunkSize;
      console.log(`Chunk uploaded: ${(offset / 1024 / 1024).toFixed(1)}/${(fileSize / 1024 / 1024).toFixed(1)}MB`);
    }

    fs.closeSync(fd);
    return null;
  } catch (err) {
    console.error('Gemini chunked upload error:', err);
    return null;
  }
}

// ============================================================
// GEMINI FILE STATUS + CLEANUP
// ============================================================

async function waitForFileActive(fileName: string, apiKey: string): Promise<boolean> {
  const start = Date.now();

  while (Date.now() - start < ACTIVE_WAIT_TIMEOUT_MS) {
    try {
      const resp = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/${fileName}?key=${apiKey}`,
      );
      if (resp.ok) {
        const data = await resp.json();
        if (data.state === 'ACTIVE') {
          console.log(`File ${fileName} is ACTIVE`);
          return true;
        }
        if (data.state === 'FAILED') {
          console.error(`File ${fileName} processing FAILED:`, JSON.stringify(data));
          return false;
        }
        console.log(`File ${fileName} state: ${data.state}, waiting...`);
      } else {
        await resp.text();
      }
    } catch (e) {
      console.error('Error polling file status:', e);
    }
    await new Promise((r) => setTimeout(r, 3000));
  }

  console.error(`File ${fileName} did not become ACTIVE within ${ACTIVE_WAIT_TIMEOUT_MS / 1000}s`);
  return false;
}

function deleteGeminiFile(fileName: string, apiKey: string): void {
  fetch(
    `https://generativelanguage.googleapis.com/v1beta/${fileName}?key=${apiKey}`,
    { method: 'DELETE' },
  ).then(r => r.text()).then(() => {
    console.log(`Deleted Gemini file: ${fileName}`);
  }).catch(e => {
    console.error('Failed to delete Gemini file (non-blocking):', e);
  });
}

// ============================================================
// GEMINI ANALYSIS
// ============================================================

async function analyzeWithGemini(
  fileUri: string,
  mimeType: string,
  apiKey: string,
  model: string,
  includeVisual: boolean,
): Promise<Record<string, any> | null> {
  const systemPrompt = `You are an expert sales call delivery analyst specializing in audio${includeVisual ? ' and video' : ''} analysis. Analyze this sales call recording and return a comprehensive JSON object.

You MUST analyze what you actually hear${includeVisual ? ' and see' : ''} — do not fabricate or guess.

Return this exact JSON structure:
{
  "tone": "warm/neutral/aggressive/monotone/enthusiastic",
  "pacing": "too_fast/good/too_slow/varied",
  "talk_to_listen_ratio": 0.65,
  "energy_level": "high/medium/low",
  "monologue_flags": ["2:30-4:15 rep spoke for 1m45s without pause"],
  "delivery_notes": "Brief overall delivery assessment",

  "filler_word_count": 12,
  "filler_examples": ["um at 1:23", "uh at 3:45", "like at 5:02"],
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
  "active_listening_signals": ["mmhm at 3:20", "repeated prospect's words at 4:10"],

  ${includeVisual ? `"visual_analysis": {
    "eye_contact_quality": "strong/moderate/poor",
    "body_language": "open/neutral/closed",
    "professionalism": "high/medium/low",
    "environment_notes": "description of background, lighting, camera framing",
    "notable_gestures": ["nodded emphatically at 3:45", "leaned in during pitch at 6:00"]
  },` : `"visual_analysis": null,`}

  "engagement_timeline": [
    { "minute": 1, "rep_energy": "high/medium/low", "prospect_engagement": "high/medium/low/disengaged", "notes": "brief note" }
  ],

  "critical_moments": [
    { "timestamp": "4:30", "type": "objection_raised/rapport_peak/energy_drop/closing_attempt/breakthrough", "description": "What happened", "rep_response_quality": "excellent/good/fair/poor" }
  ],

  "delivery_grade": "A/B/C/D/F",
  "delivery_summary": "2-3 sentence summary of overall delivery quality including vocal presence, emotional intelligence, and conversation control",

  "verification": {
    "speaker_count": 2,
    "first_words_spoken": "the first 8-10 words spoken in the recording",
    "approx_duration_seconds": 120,
    "speaker_names_heard": ["names of speakers you can identify from the audio"]
  }
}

IMPORTANT RULES:
- The "verification" block is critical — report what you actually hear. Do not fabricate.
- engagement_timeline should have one entry per minute of the call (up to 60 entries max).
- critical_moments should capture 3-8 key turning points.
- filler_examples: list up to 10 most notable instances with timestamps.
- emotional_shifts: only include genuine shifts you detect, not every moment.
- Only return the JSON object, no other text.`;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), GENERATE_CONTENT_TIMEOUT_MS);

    const resp = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          systemInstruction: { parts: [{ text: systemPrompt }] },
          contents: [{
            role: 'user',
            parts: [
              { text: 'Analyze the delivery of this sales call:' },
              { fileData: { mimeType, fileUri } },
            ],
          }],
          generationConfig: { maxOutputTokens: 16384, responseMimeType: 'application/json' },
        }),
      },
    );

    clearTimeout(timeout);

    if (!resp.ok) {
      const errBody = await resp.text();
      console.error('Gemini generateContent error:', resp.status, errBody);
      return null;
    }

    const data = await resp.json();
    const rawText = data.candidates?.[0]?.content?.parts?.[0]?.text || '{}';
    const result = extractJSON(rawText);
    console.log('GEMINI VERIFICATION:', JSON.stringify(result.verification || null));
    const { verification: _v, ...cleanResult } = result;
    return cleanResult;
  } catch (err) {
    console.error('Gemini analysis failed:', err);
    return null;
  }
}

// ============================================================
// AI CALL HELPER
// ============================================================

async function callAI(
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
      body: JSON.stringify({ model, max_tokens: 16384, system: systemPrompt, messages: [{ role: 'user', content: prompt }] }),
    });
    if (!resp.ok) throw new Error(`Anthropic error: ${resp.status} ${await resp.text()}`);
    const data = await resp.json();
    return data.content[0]?.text || '';
  }

  if (provider === 'openai' || provider === 'openrouter') {
    const baseUrl = provider === 'openrouter' ? 'https://openrouter.ai/api/v1' : 'https://api.openai.com/v1';
    const resp = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
      body: JSON.stringify({
        model, max_tokens: 16384,
        messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: prompt }],
        response_format: { type: 'json_object' },
      }),
    });
    if (!resp.ok) throw new Error(`${provider} error: ${resp.status} ${await resp.text()}`);
    const data = await resp.json();
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
          generationConfig: { maxOutputTokens: 16384, responseMimeType: 'application/json' },
        }),
      },
    );
    if (!resp.ok) throw new Error(`Gemini error: ${resp.status} ${await resp.text()}`);
    const data = await resp.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
  }

  throw new Error(`Unsupported provider: ${provider}`);
}

// ============================================================
// SCORING PROMPT
// ============================================================

function buildScoringPrompt(
  transcript: string,
  rubric: { steps: any[]; script_content?: string; script_sections?: any[]; max_total_score?: number },
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
      "reasoning": "string (2-3 sentences explaining the score with [mm:ss] timestamp references to specific moments — if audio/video delivery data is available, explicitly mention how delivery quality like tone, vocal confidence, energy, or visual presence affected this step's score)",
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
    "summary": "string (the single biggest reason this deal was lost or weakened — the #1 thing that cost them the sale)",
    "timestamp": "[mm:ss]",
    "quote": "exact words the rep said at that moment",
    "what_to_do_instead": "string (specific alternative behavior with example language the rep should have used)"
  },
  "coaching_fixes": [
    {
      "issue": "string (what went wrong — include [mm:ss] timestamp and exact quote)",
      "fix": "string (exact alternative behavior to practice, with example language)",
      "script_reference": "string or null (which script/rubric section this maps to, if applicable)"
    }
  ],
  "assigned_drill": {
    "name": "string (a memorable drill name like 'The 8-Second Torture', 'The Mirror Close', 'The Silence Challenge')",
    "why": "string (which metric was weakest and exactly why this drill fixes it — reference specific data)"
  },
  "coaching_notes": "string (overall narrative coaching summary — be direct and specific about what happened and why the score is what it is)",
  "strengths": ["string"],
  "improvements": ["string"]
}

CRITICAL SCORING RULES:
1. TIMESTAMPS REQUIRED: Every piece of evidence, every quote, every reasoning reference MUST include [mm:ss] timestamps. Never cite a moment without its timestamp.
2. SILENCE DISCIPLINE: If audio data shows talk_to_listen_ratio > 0.45 (rep talking >45%), penalize accordingly. If strategic_pauses = 0 after close attempts or price reveals, this is a major failure — flag it.
3. PRE-DISQUALIFICATION DETECTION: If the rep objects FOR the prospect (e.g., "I don't think you can afford this" or "my hesitation would be..."), this is a critical coaching issue. Flag it in deal_killer or coaching_fixes.
4. DEAL KILLER: Always identify the single biggest moment that cost the deal or weakened the outcome. Be brutally honest.
5. COACHING FIXES: Provide 2-4 specific, actionable fixes with timestamps. Each fix must have a concrete "do this instead" with example language.
6. ASSIGNED DRILL: Based on the weakest metric (silence discipline, objection handling, discovery depth, close technique), assign a specific practice drill.

Score fairly and specifically. Use exact transcript quotes as evidence. Be direct — not generic.

CRITICAL OUTPUT FORMAT: Return ONLY the raw JSON object. Do NOT include markdown headers, code fences, explanatory text, or any content before or after the JSON. Your entire response must be valid JSON starting with { and ending with }.`;

  if (rubric.script_content) {
    system += `\n\nThe rep should follow this sales script. Measure how closely they adhered to each section:
--- SALES SCRIPT ---
${rubric.script_content}
--- END SCRIPT ---`;
  }

  if (rubric.script_sections && rubric.script_sections.length > 0) {
    system += `\n\nScript section-to-rubric mapping with key phrases to look for:
${JSON.stringify(rubric.script_sections, null, 2)}

For each mapped section, calculate adherence_percentage (0-100) and track key_phrases_hit/key_phrases_missed.`;
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
    system += `\n\nIMPORTANT: Audio/video delivery analysis has been performed on this call. Factor these insights into your scoring:
- A rep who says the right words but delivers them poorly (low confidence, flat energy, no rapport) should score LOWER.
- Strong delivery can elevate scores for steps where the rep showed genuine engagement and authority.
- SILENCE DISCIPLINE: Check strategic_pauses count. Target: at least 1-2 pauses after close attempts and price reveals. If strategic_pauses = 0, this is a critical failure — mention it in deal_killer or coaching_fixes.
- TALK RATIO: If talk_to_listen_ratio > 0.45 (rep talking >45% of the time), the rep is talking too much. Flag this and penalize discovery/rapport steps.
- PRE-DISQUALIFICATION: Watch for moments where the rep objects on behalf of the prospect or downsells before the prospect raises an objection. This is a deal-killing behavior.
- Use the critical_moments and emotional_shifts data to identify the exact turning points in the call.`;

    user += `\n\n## Audio/Video Delivery Analysis
Delivery Grade: ${audioInsights.delivery_grade || 'N/A'}
Vocal Confidence: ${audioInsights.vocal_confidence || 'N/A'}
Tone: ${audioInsights.tone || 'N/A'}
Energy: ${audioInsights.energy_level || 'N/A'}
Rapport Quality: ${audioInsights.rapport_quality || 'N/A'}
Filler Words: ${audioInsights.filler_word_count ?? 'N/A'}
Enthusiasm: ${audioInsights.enthusiasm_level || 'N/A'}
Interruptions (rep/prospect): ${audioInsights.interruption_count ? `${audioInsights.interruption_count.rep}/${audioInsights.interruption_count.prospect}` : 'N/A'}
Strategic Pauses: ${audioInsights.strategic_pauses ?? 'N/A'}
Delivery Summary: ${audioInsights.delivery_summary || audioInsights.delivery_notes || 'N/A'}`;

    if (audioInsights.critical_moments?.length) {
      user += `\n\nCritical Moments:\n${audioInsights.critical_moments.map((m: any) => `- [${m.timestamp}] ${m.type}: ${m.description} (rep response: ${m.rep_response_quality})`).join('\n')}`;
    }

    if (audioInsights.emotional_shifts?.length) {
      user += `\n\nEmotional Shifts:\n${audioInsights.emotional_shifts.map((s: any) => `- [${s.timestamp}] ${s.shift} — trigger: ${s.trigger}`).join('\n')}`;
    }

    if (audioInsights.visual_analysis) {
      const v = audioInsights.visual_analysis;
      user += `\n\nVisual Analysis: Eye contact: ${v.eye_contact_quality}, Body language: ${v.body_language}, Professionalism: ${v.professionalism}`;
      if (v.environment_notes) user += `, Environment: ${v.environment_notes}`;
    }
  }

  user += `\n\n## Transcript\n${truncated}`;

  return { system, user };
}

// ============================================================
// JSON EXTRACTOR
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
  console.warn('extractJSON: first parse failed, attempting repair...');
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
    console.log('extractJSON: repair successful');
    return result;
  } catch (e) {
    console.error('extractJSON: repair failed. First 500 chars:', text.substring(0, 500));
    throw new Error(`Failed to parse AI response as JSON after repair. First 300 chars: ${raw.substring(0, 300)}`);
  }
}
