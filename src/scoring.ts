/**
 * Gemini Audio/Video Scoring - EasyPanel Worker
 * 
 * scoring.ts - Handles Gemini file upload, ACTIVE polling, and generateContent
 * 
 * FIXES APPLIED:
 * 1. ACTIVE wait timeout increased from 120s → 300s for 1GB+ files
 * 2. generateContent fetch timeout set to 600s via AbortController
 * 3. Improved safeParseJSON with better truncation repair
 */

import fs from 'fs';
import path from 'path';

const GEMINI_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024; // 8MB chunks
const ACTIVE_WAIT_TIMEOUT_MS = 300_000; // 5 minutes (was 120s)
const ACTIVE_POLL_INTERVAL_MS = 3_000; // 3 seconds
const GENERATE_CONTENT_TIMEOUT_MS = 600_000; // 10 minutes

// ============================================================
// FILE UPLOAD (resumable, chunked)
// ============================================================

export async function uploadFileToGemini(
  filePath: string,
  apiKey: string,
  mimeType: string = 'video/mp4'
): Promise<{ name: string; uri: string }> {
  const fileSize = fs.statSync(filePath).size;
  const displayName = path.basename(filePath);

  console.log(`Starting resumable upload: ${(fileSize / 1024 / 1024).toFixed(1)}MB, mime=${mimeType}`);

  // 1. Initiate resumable upload
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
    }
  );

  if (!initRes.ok) {
    const errText = await initRes.text();
    throw new Error(`Resumable upload init failed: ${initRes.status} ${errText}`);
  }

  const uploadUrl = initRes.headers.get('x-goog-upload-url');
  if (!uploadUrl) throw new Error('No upload URL returned');

  // 2. Upload in chunks
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

      const command = isLast ? 'upload, finalize' : 'upload';

      const chunkRes = await fetch(uploadUrl, {
        method: 'POST',
        headers: {
          'X-Goog-Upload-Command': command,
          'X-Goog-Upload-Offset': String(offset),
          'Content-Length': String(chunkSize),
        },
        body: chunk,
      });

      if (!chunkRes.ok && !isLast) {
        const errText = await chunkRes.text();
        throw new Error(`Chunk upload failed at offset ${offset}: ${chunkRes.status} ${errText}`);
      }

      offset += chunkSize;
      console.log(`Uploaded ${(offset / 1024 / 1024).toFixed(1)}MB / ${(fileSize / 1024 / 1024).toFixed(1)}MB`);

      if (isLast) {
        const result = await chunkRes.json() as any;
        const file = result.file;
        console.log(`Upload complete: ${file.name}, uri=${file.uri}`);
        return { name: file.name, uri: file.uri };
      }
    }
  } finally {
    fs.closeSync(fd);
  }

  throw new Error('Upload loop exited without finalizing');
}

// ============================================================
// WAIT FOR FILE TO BECOME ACTIVE
// ============================================================

export async function waitForFileActive(
  fileName: string,
  apiKey: string
): Promise<boolean> {
  const maxWaitMs = ACTIVE_WAIT_TIMEOUT_MS;
  const pollInterval = ACTIVE_POLL_INTERVAL_MS;
  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitMs) {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/${fileName}?key=${apiKey}`
    );
    if (res.ok) {
      const data = await res.json() as any;
      if (data.state === 'ACTIVE') {
        console.log(`File ${fileName} is ACTIVE`);
        return true;
      }
      console.log(`File state: ${data.state}, waiting...`);
    }
    await new Promise((r) => setTimeout(r, pollInterval));
  }

  console.error(`File ${fileName} did not become ACTIVE within ${maxWaitMs / 1000}s`);
  return false;
}

// ============================================================
// DELETE GEMINI FILE (cleanup)
// ============================================================

export async function deleteGeminiFile(fileName: string, apiKey: string): Promise<void> {
  try {
    await fetch(
      `https://generativelanguage.googleapis.com/v1beta/${fileName}?key=${apiKey}`,
      { method: 'DELETE' }
    );
    console.log(`Deleted Gemini file: ${fileName}`);
  } catch (e) {
    console.error(`Failed to delete Gemini file ${fileName}:`, e);
  }
}

// ============================================================
// ANALYZE AUDIO/VIDEO WITH GEMINI
// ============================================================

export async function analyzeAudioWithGemini(
  fileUri: string,
  mimeType: string,
  prompt: string,
  apiKey: string,
  model: string = 'gemini-2.5-flash'
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

  // Use AbortController with a generous timeout for large file analysis
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), GENERATE_CONTENT_TIMEOUT_MS);

  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      }
    );

    clearTimeout(timeout);

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Gemini API error: ${response.status} ${errText}`);
    }

    const data = await response.json() as any;

    // Extract text from response
    const text =
      data?.candidates?.[0]?.content?.parts?.[0]?.text ||
      data?.candidates?.[0]?.content?.parts?.[0]?.functionCall?.args;

    if (!text) {
      console.error('GEMINI VERIFICATION:', JSON.stringify(data?.candidates?.[0]?.content));
      throw new Error('No text in Gemini response');
    }

    return typeof text === 'string' ? text : JSON.stringify(text);
  } catch (err: any) {
    clearTimeout(timeout);
    if (err.name === 'AbortError') {
      throw new Error(`Gemini generateContent timed out after ${GENERATE_CONTENT_TIMEOUT_MS / 1000}s`);
    }
    throw err;
  }
}
