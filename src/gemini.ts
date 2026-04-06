import * as fs from 'fs';
import * as path from 'path';
import { pipeline } from 'stream/promises';

const CHUNK_SIZE = 8 * 1024 * 1024; // 8 MB

// ── Download recording to disk ──

export async function downloadToTmp(url: string, jobId: string): Promise<{ filePath: string; size: number }> {
  const ext = url.split('?')[0].split('.').pop()?.toLowerCase() || 'mp4';
  const filePath = path.join('/tmp', `scoring-${jobId}.${ext}`);

  const resp = await fetch(url);
  if (!resp.ok || !resp.body) throw new Error(`Download failed: ${resp.status}`);

  const fileStream = fs.createWriteStream(filePath);
  // @ts-ignore - ReadableStream to Node stream
  await pipeline(resp.body as any, fileStream);

  const stats = fs.statSync(filePath);
  console.log(`Downloaded ${(stats.size / 1024 / 1024).toFixed(1)}MB to ${filePath}`);
  return { filePath, size: stats.size };
}

// ── Resumable chunked upload to Gemini File API ──

export async function uploadToGemini(
  filePath: string,
  apiKey: string,
  mimeType: string,
): Promise<{ fileUri: string; fileName: string }> {
  const stats = fs.statSync(filePath);
  const totalSize = stats.size;

  console.log(`Starting resumable upload: ${(totalSize / 1024 / 1024).toFixed(1)}MB, mime=${mimeType}`);

  // Step 1: Initiate resumable upload
  const initResp = await fetch(
    `https://generativelanguage.googleapis.com/upload/v1beta/files?key=${apiKey}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Goog-Upload-Protocol': 'resumable',
        'X-Goog-Upload-Command': 'start',
        'X-Goog-Upload-Header-Content-Length': totalSize.toString(),
        'X-Goog-Upload-Header-Content-Type': mimeType,
      },
      body: JSON.stringify({ file: { display_name: path.basename(filePath) } }),
    },
  );

  if (!initResp.ok) {
    const err = await initResp.text();
    throw new Error(`Gemini resumable init failed (${initResp.status}): ${err}`);
  }

  const uploadUrl = initResp.headers.get('x-goog-upload-url');
  if (!uploadUrl) throw new Error('No upload URL returned from Gemini');

  // Step 2: Upload chunks
  const fd = fs.openSync(filePath, 'r');
  let offset = 0;

  try {
    while (offset < totalSize) {
      const remaining = totalSize - offset;
      const chunkSize = Math.min(CHUNK_SIZE, remaining);
      const isLast = offset + chunkSize >= totalSize;

      const buffer = Buffer.alloc(chunkSize);
      fs.readSync(fd, buffer, 0, chunkSize, offset);

      const command = isLast ? 'upload, finalize' : 'upload';

      const chunkResp = await fetch(uploadUrl, {
        method: 'PUT',
        headers: {
          'Content-Length': chunkSize.toString(),
          'X-Goog-Upload-Offset': offset.toString(),
          'X-Goog-Upload-Command': command,
        },
        body: buffer,
      });

      if (!chunkResp.ok && !isLast) {
        const errText = await chunkResp.text();
        throw new Error(`Chunk upload failed at offset ${offset}: ${chunkResp.status} ${errText}`);
      }

      if (isLast) {
        const result: any = await chunkResp.json();
        const file = result.file;
        if (!file?.uri || !file?.name) {
          throw new Error(`Gemini upload finalize returned unexpected: ${JSON.stringify(result)}`);
        }
        console.log(`Upload complete: ${file.name}, uri=${file.uri}`);
        return { fileUri: file.uri, fileName: file.name };
      }

      offset += chunkSize;
      console.log(`Uploaded ${(offset / 1024 / 1024).toFixed(1)}MB / ${(totalSize / 1024 / 1024).toFixed(1)}MB`);
    }
  } finally {
    fs.closeSync(fd);
  }

  throw new Error('Upload loop ended without finalizing');
}

// ── Wait for file to become ACTIVE ──

export async function waitForActive(fileName: string, apiKey: string, maxWaitMs = 120_000): Promise<boolean> {
  const start = Date.now();
  const pollMs = 3000;

  while (Date.now() - start < maxWaitMs) {
    try {
      const resp = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/${fileName}?key=${apiKey}`,
      );
      if (resp.ok) {
        const data: any = await resp.json();
        if (data.state === 'ACTIVE') {
          console.log(`File ${fileName} is ACTIVE`);
          return true;
        }
        if (data.state === 'FAILED') {
          console.error(`File ${fileName} FAILED:`, JSON.stringify(data));
          return false;
        }
        console.log(`File state: ${data.state}, waiting...`);
      } else {
        await resp.text();
      }
    } catch (e) {
      console.error('Poll error:', e);
    }
    await new Promise(r => setTimeout(r, pollMs));
  }

  console.error(`File did not become ACTIVE within ${maxWaitMs / 1000}s`);
  return false;
}

// ── Delete file from Gemini ──

export async function deleteGeminiFile(fileName: string, apiKey: string): Promise<void> {
  try {
    const resp = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/${fileName}?key=${apiKey}`,
      { method: 'DELETE' },
    );
    await resp.text();
    console.log(`Deleted Gemini file: ${fileName}`);
  } catch (e) {
    console.error('Failed to delete Gemini file:', e);
  }
}

// ── Cleanup temp file ──

export function cleanupTmpFile(filePath: string): void {
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      console.log(`Cleaned up: ${filePath}`);
    }
  } catch (e) {
    console.error('Cleanup error:', e);
  }
}

// ── Detect MIME type from URL ──

export function detectMimeType(url: string): string {
  const ext = url.split('?')[0].split('.').pop()?.toLowerCase() || '';
  const mimeMap: Record<string, string> = {
    mp4: 'video/mp4', webm: 'video/webm', mov: 'video/quicktime',
    mp3: 'audio/mpeg', wav: 'audio/wav', m4a: 'audio/mp4',
    ogg: 'audio/ogg', flac: 'audio/flac',
  };
  return mimeMap[ext] || 'video/mp4';
}
