/**
 * Scoring Worker — Main entry point
 *
 * Polls pgmq scoring_jobs queue and processes large recording scoring jobs.
 * Designed for EasyPanel deployment.
 */

import { readQueue, deleteMessage, updateCallScore, deleteScoreDetails, insertScoreDetails, shutdown } from './db';
import { startHealthServer, recordPoll, recordJobDone, recordJobFailed } from './health';
import { analyzeAudioWithGemini, buildScoringPrompt, callAI } from './scoring';

const POLL_INTERVAL = parseInt(process.env.POLL_INTERVAL_MS || '5000', 10);
const VIS_TIMEOUT = parseInt(process.env.VISIBILITY_TIMEOUT_S || '300', 10);
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '3', 10);
const SUPABASE_URL = process.env.SUPABASE_URL!;
const SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

let running = true;

async function processJob(job: any): Promise<void> {
  const msg = job.message;
  const {
    call_id, sub_account_id, workspace_id, call_score_id,
    recording_url, transcript, transcript_source, recording_id,
    rubric, call_context, scored_by,
    gemini_api_key, gemini_model,
    scoring_api_key, scoring_model, scoring_provider,
  } = msg;

  console.log(`Processing job for call ${call_id} (score ${call_score_id})`);

  // Update status to processing
  await updateCallScore(call_score_id, {
    status: 'processing',
    updated_at: new Date().toISOString(),
  });

  // 1. Audio analysis (the whole reason this is in the worker)
  let audioDelivery: Record<string, any> | null = null;
  let analysisType = 'transcript_only';

  if (recording_url && gemini_api_key && transcript_source !== 'fathom') {
    audioDelivery = await analyzeAudioWithGemini(
      recording_url,
      gemini_api_key,
      gemini_model || 'gemini-2.5-flash',
      job.msg_id,
    );
    if (audioDelivery) analysisType = 'audio_and_transcript';
  }

  // 2. Score transcript
  const { system, user } = buildScoringPrompt(transcript, rubric, call_context, audioDelivery);

  const rawResult = await callAI(user, system, scoring_api_key, scoring_model, scoring_provider);
  const cleaned = rawResult.replace(/^```(?:json)?\s*\n?/i, '').replace(/\n?\s*```\s*$/i, '').trim();
  const scoringResult = JSON.parse(cleaned);

  // 3. Write results
  const overallScore = scoringResult.overall_score || 0;
  const maxPossible = scoringResult.max_possible_score || rubric.max_total_score;

  const enrichedAudioDelivery = {
    ...(audioDelivery || {}),
    deal_killer: scoringResult.deal_killer || null,
    coaching_fixes: scoringResult.coaching_fixes || [],
    assigned_drill: scoringResult.assigned_drill || null,
  };

  await updateCallScore(call_score_id, {
    rubric_id: rubric.id,
    overall_score: overallScore,
    max_possible_score: maxPossible,
    analysis_type: analysisType,
    audio_delivery: enrichedAudioDelivery,
    objections_detected: scoringResult.objections_detected || [],
    coaching_notes: scoringResult.coaching_notes || '',
    strengths: scoringResult.strengths || [],
    improvements: scoringResult.improvements || [],
    transcript_source: transcript_source,
    recording_id: recording_id,
    ai_provider: scoring_provider,
    ai_model: scoring_model,
    scored_by: scored_by,
    status: 'completed',
    scored_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    error_message: null,
  });

  // Insert step details
  if (scoringResult.step_scores && Array.isArray(scoringResult.step_scores)) {
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

  // 4. Trigger Slack alert (fire-and-forget)
  const scorePercentage = maxPossible > 0 ? Math.round((overallScore / maxPossible) * 100) : 0;

  try {
    await fetch(`${SUPABASE_URL}/functions/v1/send-slack-alert`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${SERVICE_KEY}`, 'Content-Type': 'application/json' },
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
            label: s.step_label, score: s.score || 0, max: s.max_score || 0,
          })),
          delivery_grade: audioDelivery?.delivery_grade || null,
          delivery_summary: audioDelivery?.delivery_summary || null,
          tone: audioDelivery?.tone || null,
          energy_level: audioDelivery?.energy_level || null,
          vocal_confidence: audioDelivery?.vocal_confidence || null,
          pacing: audioDelivery?.pacing || null,
          objections: (scoringResult.objections_detected || []).map((o: any) => ({
            objection: o.objection, handled: o.handled_well, quality: o.response_quality,
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

  console.log(`Call ${call_id} scored: ${scorePercentage}% (${scoring_provider}/${scoring_model})`);
}

async function pollLoop(): Promise<void> {
  console.log('Scoring worker started — polling for jobs...');

  while (running) {
    try {
      recordPoll();
      const job = await readQueue(VIS_TIMEOUT);

      if (!job) {
        await new Promise(r => setTimeout(r, POLL_INTERVAL));
        continue;
      }

      // Check retry count
      if (job.read_ct > MAX_RETRIES) {
        console.error(`Job ${job.msg_id} exceeded max retries (${job.read_ct}), marking failed`);
        const msg = job.message as any;
        if (msg?.call_score_id) {
          await updateCallScore(msg.call_score_id, {
            status: 'failed',
            error_message: `Worker exceeded max retries (${MAX_RETRIES})`,
            updated_at: new Date().toISOString(),
          });
        }
        await deleteMessage(job.msg_id);
        recordJobFailed();
        continue;
      }

      try {
        await processJob(job);
        await deleteMessage(job.msg_id);
        recordJobDone();
      } catch (err) {
        console.error(`Job ${job.msg_id} failed (attempt ${job.read_ct}):`, err);
        // Update score status to failed if final attempt
        const msg = job.message as any;
        if (job.read_ct >= MAX_RETRIES && msg?.call_score_id) {
          await updateCallScore(msg.call_score_id, {
            status: 'failed',
            error_message: `Worker error: ${err instanceof Error ? err.message : String(err)}`,
            updated_at: new Date().toISOString(),
          });
          await deleteMessage(job.msg_id);
        }
        // Otherwise let visibility timeout expire for retry
        recordJobFailed();
      }
    } catch (pollErr) {
      console.error('Poll loop error:', pollErr);
      await new Promise(r => setTimeout(r, POLL_INTERVAL * 2));
    }
  }
}

// ── Graceful shutdown ──
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down...');
  running = false;
});
process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down...');
  running = false;
});

// ── Start ──
startHealthServer();
pollLoop().then(() => {
  console.log('Worker stopped');
  shutdown();
});
