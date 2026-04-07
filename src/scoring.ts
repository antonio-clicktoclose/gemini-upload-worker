/**
 * Scoring Engine — processes a single PGMQ job
 *
 * Only change from scoring_8: analyzeAudioWithGemini uses undici.request
 * instead of fetch to avoid UND_ERR_HEADERS_TIMEOUT on large files.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { spawnSync } from 'node:child_process';
import { request as undiciRequest } from 'undici';

const GEMINI_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024;
const ACTIVE_WAIT_TIMEOUT_MS = 300_000;
const ACTIVE_POLL_INTERVAL_MS = 3_000;
const GENERATE_CONTENT_TIMEOUT_MS = 600_000;
const GEMINI_MAX_MEDIA_TOKENS = 1_000_000;
const TOKENS_PER_SECOND = 290;

type GeminiFileResource = {
  name?: string;
  uri?: string;
  state?: string;
};

type GeminiUploadResponse = {
  file?: GeminiFileResource;
};

type GeminiStatusResponse = GeminiFileResource;

type GeminiGenerateResponse = {
  candidates?: Array<{
    content?: {
      parts?: Array<{
        text?: string;
        functionCall?: {
          args?: unknown;
        };
      }>;
    };
  }>;
};

async function readJson<T>(response: Response): Promise<T> {
  return (await response.json()) as T;
}

export function estimateGeminiTokens(durationSeconds: number): number {
  return Math.ceil(durationSeconds * TOKENS_PER_SECOND);
}

export function exceedsGeminiMediaTokenLimit(durationSeconds: number): boolean {
  return estimateGeminiTokens(durationSeconds) > GEMINI_MAX_MEDIA_TOKENS;
}

export function getMediaDurationSeconds(filePath: string): number | null {
  const result = spawnSync(
    'ffprobe',
    [
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=noprint_wrappers=1:nokey=1',
      filePath,
    ],
    { encoding: 'utf8' },
  );
  if (result.status !== 0) {
    console.error('ffprobe failed:', result.stderr || result.stdout);
    return null;
  }
  const duration = Number.parseFloat((result.stdout || '').trim());
  return Number.isFinite(duration) ? duration : null;
}

export function extractAudioToOgg(inputFilePath: string, outputFilePath: string): void {
  const result = spawnSync(
    'ffmpeg',
    ['-y', '-i', inputFilePath, '-vn', '-acodec', 'libopus', '-b:a', '48k', outputFilePath],
    { encoding: 'utf8' },
  );
  if (result.status !== 0) {
    throw new Error(`ffmpeg extraction failed: ${result.stderr || result.stdout || 'unknown error'}`);
  }
  if (!fs.existsSync(outputFilePath)) {
    throw new Error('ffmpeg reported success but no audio file was created');
  }
}

export async function uploadFileToGemini(
  filePath: string,
  apiKey: string,
  mimeType: string = 'video/mp4',
): Promise<{ name: string; uri: string }> {
  const fileSize = fs.statSync(filePath).size;
  const displayName = path.basename(filePath);
  console.log(`Starting Gemini upload: ${(fileSize / 1024 / 1024).toFixed(1)}MB, mime=${mimeType}`);

  const initRes = await fetch(
    `https://generativelanguage.googleapis.com/upload/v1beta/files?key=${apiKey}`,
    {
      method: 'POST',
      headers: {
        'X-Goog-Upload-Protocol': 'resumable',
        'X-Goog-Upload-Command': 'start',
        'X-Goog-Upload-Header-Content-Length': String(fileSize),
        'X-Goog-Upload-Header-Content-Type': mimeType,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ file: { display_name: displayName } }),
    },
  );

  if (!initRes.ok) {
    throw new Error(`Resumable upload init failed: ${initRes.status} ${await initRes.text()}`);
  }

  const uploadUrl = initRes.headers.get('x-goog-upload-url');
  if (!uploadUrl) {
    throw new Error('Gemini did not return an upload URL');
  }

  const fd = fs.openSync(filePath, 'r');
  const buffer = Buffer.alloc(GEMINI_UPLOAD_CHUNK_SIZE);
  let offset = 0;

  try {
    while (offset < fileSize) {
      const remaining = fileSize - offset;
      const chunkSize = Math.min(GEMINI_UPLOAD_CHUNK_SIZE, remaining);
      const isLast = offset + chunkSize >= fileSize;

      fs.readSync(fd, buffer, 0, chunkSize, offset);
      const chunk = buffer.subarray(0, chunkSize);

      const chunkRes = await fetch(uploadUrl, {
        method: 'POST',
        headers: {
          'X-Goog-Upload-Command': isLast ? 'upload, finalize' : 'upload',
          'X-Goog-Upload-Offset': String(offset),
          'Content-Length': String(chunkSize),
        },
        body: chunk,
      });

      if (!chunkRes.ok) {
        throw new Error(`Chunk upload failed at offset ${offset}: ${chunkRes.status} ${await chunkRes.text()}`);
      }

      offset += chunkSize;
      console.log(`Uploaded ${(offset / 1024 / 1024).toFixed(1)}MB / ${(fileSize / 1024 / 1024).toFixed(1)}MB`);

      if (isLast) {
        const result = await readJson<GeminiUploadResponse>(chunkRes);
        const file = result.file;
        if (!file?.name || !file.uri) {
          throw new Error('Gemini upload finalized without file metadata');
        }
        console.log(`Upload complete: ${file.name}`);
        return { name: file.name, uri: file.uri };
      }
    }
  } finally {
    fs.closeSync(fd);
  }

  throw new Error('Upload loop exited without returning a finalized Gemini file');
}

export async function waitForFileActive(fileName: string, apiKey: string): Promise<boolean> {
  const startTime = Date.now();

  while (Date.now() - startTime < ACTIVE_WAIT_TIMEOUT_MS) {
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/${fileName}?key=${apiKey}`);
    if (res.ok) {
      const data = await readJson<GeminiStatusResponse>(res);
      if (data.state === 'ACTIVE') {
        console.log(`Gemini file ${fileName} is ACTIVE`);
        return true;
      }
      if (data.state === 'FAILED') {
        throw new Error(`Gemini file ${fileName} entered FAILED state`);
      }
      console.log(`Gemini file ${fileName} state: ${data.state || 'unknown'}; waiting...`);
    } else {
      console.error(`Gemini file poll failed: ${res.status} ${await res.text()}`);
    }
    await new Promise((resolve) => setTimeout(resolve, ACTIVE_POLL_INTERVAL_MS));
  }

  console.error(`Gemini file ${fileName} did not become ACTIVE within ${ACTIVE_WAIT_TIMEOUT_MS / 1000}s`);
  return false;
}

export async function deleteGeminiFile(fileName: string, apiKey: string): Promise<void> {
  try {
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/${fileName}?key=${apiKey}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      console.error(`Failed to delete Gemini file ${fileName}: ${res.status} ${await res.text()}`);
      return;
    }
    console.log(`Deleted Gemini file: ${fileName}`);
  } catch (error) {
    console.error(`Failed to delete Gemini file ${fileName}:`, error);
  }
}

/**
 * analyzeAudioWithGemini — uses undici.request instead of fetch
 * to avoid UND_ERR_HEADERS_TIMEOUT on large files (>5min processing).
 */
export async function analyzeAudioWithGemini(
  fileUri: string,
  mimeType: string,
  prompt: string,
  apiKey: string,
  model: string = 'gemini-2.5-flash',
): Promise<string> {
  const requestBody = {
    contents: [
      {
        parts: [
          { file_data: { mime_type: mimeType, file_uri: fileUri } },
          { text: prompt },
        ],
      },
    ],
    generationConfig: {
      temperature: 0.3,
      maxOutputTokens: 16384,
      responseMimeType: 'application/json',
    },
  };

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

  const resp = await undiciRequest(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
    headersTimeout: GENERATE_CONTENT_TIMEOUT_MS,  // 10 min — prevents UND_ERR_HEADERS_TIMEOUT
    bodyTimeout: GENERATE_CONTENT_TIMEOUT_MS,      // 10 min
  });

  if (resp.statusCode !== 200) {
    const errBody = await resp.body.text();
    throw new Error(`Gemini API error: ${resp.statusCode} ${errBody}`);
  }

  const data = (await resp.body.json()) as GeminiGenerateResponse;
  const firstPart = data.candidates?.[0]?.content?.parts?.[0];
  const directText = firstPart?.text;
  const functionArgs = firstPart?.functionCall?.args;

  if (typeof directText === 'string' && directText.trim()) {
    return directText;
  }
  if (typeof functionArgs === 'string' && functionArgs.trim()) {
    return functionArgs;
  }
  if (functionArgs !== undefined) {
    return JSON.stringify(functionArgs);
  }

  console.error('Unexpected Gemini response:', JSON.stringify(data));
  throw new Error('No usable text payload in Gemini response');
}
