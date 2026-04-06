import { Pool, PoolClient } from 'pg';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 5,
  idleTimeoutMillis: 30000,
});

export async function query(text: string, params?: any[]) {
  const client = await pool.connect();
  try {
    return await client.query(text, params);
  } finally {
    client.release();
  }
}

export async function getClient(): Promise<PoolClient> {
  return pool.connect();
}

// ── PostgreSQL array literal helper ──
function toPgArray(arr: any[] | null | undefined): string | null {
  if (!arr || arr.length === 0) return '{}';
  const escaped = arr.map(s =>
    '"' + String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"'
  );
  return '{' + escaped.join(',') + '}';
}

// ── pgmq helpers ──

export async function readQueue(visibilityTimeout: number): Promise<any | null> {
  const result = await query(
    `SELECT * FROM pgmq.read('scoring_jobs'::text, $1::integer, 1::integer)`,
    [visibilityTimeout]
  );
  return result.rows.length > 0 ? result.rows[0] : null;
}

export async function deleteMessage(msgId: number): Promise<void> {
  await query(
    `SELECT pgmq.delete('scoring_jobs'::text, $1::bigint)`,
    [msgId]
  );
}

// ── Score persistence ──

export async function updateCallScore(callScoreId: string, updates: Record<string, any>): Promise<void> {
  const keys = Object.keys(updates);
  const sets = keys.map((k, i) => {
    if (['audio_delivery', 'objections_detected', 'strengths', 'improvements'].includes(k)) {
      return `"${k}" = $${i + 1}::jsonb`;
    }
    return `"${k}" = $${i + 1}`;
  });
  const values = keys.map(k => {
    const v = updates[k];
    if (typeof v === 'object' && v !== null && !Array.isArray(v)) return JSON.stringify(v);
    if (Array.isArray(v)) return JSON.stringify(v);
    return v;
  });
  values.push(callScoreId);
  await query(
    `UPDATE public.call_scores SET ${sets.join(', ')} WHERE id = $${values.length}`,
    values
  );
}

export async function deleteScoreDetails(callScoreId: string): Promise<void> {
  await query(`DELETE FROM public.call_score_details WHERE call_score_id = $1`, [callScoreId]);
}

export async function insertScoreDetails(details: any[]): Promise<void> {
  if (details.length === 0) return;

  const columns = [
    'call_score_id', 'step_key', 'step_label', 'score', 'max_score',
    'reasoning', 'evidence', 'adherence_percentage',
    'key_phrases_hit', 'key_phrases_missed'
  ];

  const valuePlaceholders: string[] = [];
  const values: any[] = [];
  let paramIdx = 1;

  for (const d of details) {
    const placeholders = columns.map(() => `$${paramIdx++}`);
    valuePlaceholders.push(`(${placeholders.join(', ')})`);
    values.push(
      d.call_score_id, d.step_key, d.step_label, d.score, d.max_score,
      d.reasoning || '',
      toPgArray(d.evidence),
      d.adherence_percentage ?? null,
      toPgArray(d.key_phrases_hit),
      toPgArray(d.key_phrases_missed)
    );
  }

  await query(
    `INSERT INTO public.call_score_details (${columns.join(', ')}) VALUES ${valuePlaceholders.join(', ')}`,
    values
  );
}

export async function shutdown(): Promise<void> {
  await pool.end();
}
