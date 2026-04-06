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

// ── pgmq helpers ──

export interface QueueMessage {
  msg_id: string;
  read_ct: number;
  enqueued_at: string;
  vt: string;
  message: any;
}

export async function readQueue(visibilityTimeoutS: number): Promise<QueueMessage | null> {
  const result = await query(
    `SELECT * FROM pgmq.read('scoring_jobs', $1, 1)`,
    [visibilityTimeoutS]
  );
  return result.rows.length > 0 ? result.rows[0] : null;
}

export async function deleteMessage(msgId: string): Promise<void> {
  await query(`SELECT pgmq.delete('scoring_jobs', $1)`, [BigInt(msgId)]);
}

export async function archiveMessage(msgId: string): Promise<void> {
  await query(`SELECT pgmq.archive('scoring_jobs', $1)`, [BigInt(msgId)]);
}

// ── call_scores / call_score_details writes ──

export async function updateCallScore(callScoreId: string, data: Record<string, any>): Promise<void> {
  const keys = Object.keys(data);
  const sets = keys.map((k, i) => `"${k}" = $${i + 2}`).join(', ');
  const values = keys.map(k => {
    const v = data[k];
    return typeof v === 'object' && v !== null ? JSON.stringify(v) : v;
  });
  await query(`UPDATE public.call_scores SET ${sets} WHERE id = $1`, [callScoreId, ...values]);
}

export async function deleteScoreDetails(callScoreId: string): Promise<void> {
  await query(`DELETE FROM public.call_score_details WHERE call_score_id = $1`, [callScoreId]);
}

export async function insertScoreDetails(details: Record<string, any>[]): Promise<void> {
  if (details.length === 0) return;

  const columns = [
    'call_score_id', 'step_key', 'step_label', 'score', 'max_score',
    'reasoning', 'evidence', 'adherence_percentage', 'key_phrases_hit', 'key_phrases_missed'
  ];

  const valuePlaceholders: string[] = [];
  const values: any[] = [];
  let idx = 1;

  for (const d of details) {
    const placeholders = columns.map(() => `$${idx++}`);
    valuePlaceholders.push(`(${placeholders.join(', ')})`);
    values.push(
      d.call_score_id, d.step_key, d.step_label, d.score, d.max_score,
      d.reasoning || '', JSON.stringify(d.evidence || []),
      d.adherence_percentage ?? null,
      d.key_phrases_hit ? JSON.stringify(d.key_phrases_hit) : null,
      d.key_phrases_missed ? JSON.stringify(d.key_phrases_missed) : null
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
