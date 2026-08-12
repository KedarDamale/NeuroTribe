# NeuroTRIBE-HBN — Astro + React frontend.

FROM node:22-alpine AS deps
WORKDIR /app
COPY apps/web/package.json apps/web/package-lock.json* ./
RUN npm install --no-audit --no-fund


FROM node:22-alpine AS build
WORKDIR /app
ENV NODE_ENV=production
COPY --from=deps /app/node_modules ./node_modules
COPY apps/web/ ./
# PUBLIC_API_BASE is baked in at build time for the static client bundle.
ARG PUBLIC_API_BASE=/api
ENV PUBLIC_API_BASE=$PUBLIC_API_BASE
RUN npm run build


FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=4321
RUN apk add --no-cache wget
COPY --from=build /app/dist ./dist
COPY --from=deps /app/node_modules ./node_modules
COPY apps/web/package.json ./

EXPOSE 4321
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD wget -qO- http://localhost:4321/ >/dev/null 2>&1 || exit 1

CMD ["node", "./dist/server/entry.mjs"]


# --------------------------------------------------------------------------
FROM node:22-alpine AS dev
WORKDIR /app
ENV HOST=0.0.0.0
COPY --from=deps /app/node_modules ./node_modules
COPY apps/web/ ./
EXPOSE 4321
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
