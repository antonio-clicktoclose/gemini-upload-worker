FROM node:20

WORKDIR /app

COPY package.json ./
RUN npm install --production

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 3000

CMD ["npx","tsx","worker.ts"]
