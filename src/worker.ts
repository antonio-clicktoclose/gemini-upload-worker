/**
 * External Scoring Worker
 * 
 * Polls PGMQ 'scoring_jobs' queue and processes large recordings
 * that exceed the 80MB Edge Function limit.
 * 
 * Resources: 1 vCPU, 1GB RAM, 10GB SSD on EasyPanel
 * Uses disk-streaming + 8MB upload chunks to stay under ~100MB memory.
 */

import express from 'express';
import { createClient } from '@supabase/supabase-js';
import { processJob } from './scoring.js';

// ── Config ──
const SUPABASE_URL = process.env.SUPABASE_URL!;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const POLL_INTERVAL_MS = parseInt(process.env.POLL_INTERVAL_MS || '5000', 10);
const VISIBILITY_TIMEOUT = parseInt(process.env.VISIBILITY_TIMEOUT || '600', 10); // 10 min
const PORT = parseInt(process.env.PORT || '3000', 10);

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// ── Health endpoint ──
const app = express();
app.get('/health', (_req, res) => {
  res.json({ status: 'ok', uptime: process.uptime(), timestamp: new Date().toISOString() });
});
app.listen(PORT, () => console.log(`Health endpoint on :${PORT}`));

// ── Main poll loop ──
let processing = false;

async function poll() {
  if (processing) return;
  processing = true;

  try {
    const { data: messages, error } = await supabase.rpc('pgmq_read', {
      queue_name: 'scoring_jobs',
      vt: VISIBILITY_TIMEOUT,
      qty: 1,
    });

    if (error) {
      console.error('pgmq_read error:', error.message);
      return;
    }

    if (!messages || messages.length === 0) return;

    const msg = messages[0];
    const job = msg.message;
    const msgId = msg.msg_id;

    console.log(`Processing job msg_id=${msgId} call_id=${job.call_id} mode=${job.media_analysis_mode || 'video_and_audio'}`);

    try {
      await processJob(supabase, job);

      // Archive on success
      await supabase.rpc('pgmq_archive', { queue_name: 'scoring_jobs', msg_id: msgId });
      console.log(`Job msg_id=${msgId} archived successfully`);
    } catch (err) {
      console.error(`Job msg_id=${msgId} failed:`, err instanceof Error ? err.message : err);

      // Mark score as failed
      if (job.call_score_id) {
        await supabase.from('call_scores').update({
          status: 'failed',
          error_message: `Worker error: ${err instanceof Error ? err.message : String(err)}`,
          updated_at: new Date().toISOString(),
        }).eq('id', job.call_score_id);
      }

      // Delete from queue so it doesn't retry forever
      await supabase.rpc('pgmq_delete', { queue_name: 'scoring_jobs', msg_id: msgId });
    }
  } catch (e) {
    console.error('Poll loop error:', e);
  } finally {
    processing = false;
  }
}

setInterval(poll, POLL_INTERVAL_MS);
console.log(`Worker started — polling every ${POLL_INTERVAL_MS}ms`);
