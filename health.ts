import express from 'express';

const app = express();
const port = parseInt(process.env.PORT || '3000', 10);

let lastPollAt = Date.now();
let jobsProcessed = 0;
let jobsFailed = 0;

export function recordPoll() {
  lastPollAt = Date.now();
}

export function recordJobDone() {
  jobsProcessed++;
}

export function recordJobFailed() {
  jobsFailed++;
}

app.get('/health', (_req, res) => {
  const staleSec = (Date.now() - lastPollAt) / 1000;
  const healthy = staleSec < 60; // unhealthy if no poll in 60s
  res.status(healthy ? 200 : 503).json({
    status: healthy ? 'ok' : 'stale',
    last_poll_seconds_ago: Math.round(staleSec),
    jobs_processed: jobsProcessed,
    jobs_failed: jobsFailed,
    uptime_seconds: Math.round(process.uptime()),
  });
});

export function startHealthServer() {
  app.listen(port, '0.0.0.0', () => {
    console.log(`Health server listening on port ${port}`);
  });
}
