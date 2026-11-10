FROM node:22-alpine

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .

EXPOSE 5317
CMD ["npx", "vite", "--host", "0.0.0.0"]
