FROM node:20-alpine

# Install ffmpeg for audio extraction (audio_only mode)
RUN apk add --no-cache ffmpeg

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --production

COPY . .

EXPOSE 3000

CMD ["node", "worker.js"]
