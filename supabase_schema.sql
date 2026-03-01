-- CoastGen Supabase Schema
-- Run this in Supabase SQL Editor

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    email_verified INT DEFAULT 0,
    name TEXT,
    avatar_url TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- 2. OAuth Identities Table
CREATE TABLE IF NOT EXISTS oauth_identities (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    provider_user_id TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(id),
    access_token TEXT,
    refresh_token TEXT,
    token_expires_at INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CONSTRAINT unique_provider_user UNIQUE (provider, provider_user_id)
);

CREATE INDEX IF NOT EXISTS idx_oauth_user ON oauth_identities(user_id);

-- 3. Usage Events Table (for quota tracking)
CREATE TABLE IF NOT EXISTS usage_events (
    id TEXT PRIMARY KEY,
    job_id TEXT,
    principal_type TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    bucket TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_usage_principal ON usage_events(principal_type, principal_id, bucket, created_at);

-- 4. Subscriptions Table
CREATE TABLE IF NOT EXISTS subscriptions (
    user_id TEXT PRIMARY KEY REFERENCES users(id),
    provider TEXT NOT NULL,
    customer_id TEXT,
    subscription_id TEXT,
    status TEXT NOT NULL,
    period_start TEXT,
    period_end TEXT,
    plan_code TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sub_status ON subscriptions(status, period_end);

-- 5. Processed Webhooks Table (idempotency)
CREATE TABLE IF NOT EXISTS processed_webhooks (
    webhook_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    received_at TEXT NOT NULL
);

-- 6. Helper Function for OAuth Upsert
CREATE OR REPLACE FUNCTION upsert_oauth_identity(
    p_id TEXT, p_provider TEXT, p_provider_user_id TEXT, p_user_id TEXT,
    p_access_token TEXT, p_refresh_token TEXT, p_token_expires_at INTEGER,
    p_created_at TEXT, p_updated_at TEXT
) RETURNS VOID AS $$
BEGIN
    INSERT INTO oauth_identities
        (id, provider, provider_user_id, user_id, access_token, refresh_token,
         token_expires_at, created_at, updated_at)
    VALUES (p_id, p_provider, p_provider_user_id, p_user_id, p_access_token,
            p_refresh_token, p_token_expires_at, p_created_at, p_updated_at)
    ON CONFLICT (provider, provider_user_id) DO UPDATE SET
        user_id = EXCLUDED.user_id,
        access_token = EXCLUDED.access_token,
        refresh_token = EXCLUDED.refresh_token,
        token_expires_at = EXCLUDED.token_expires_at,
        updated_at = EXCLUDED.updated_at;
END;
$$ LANGUAGE plpgsql;

-- Enable Row Level Security (RLS) on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE oauth_identities ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE processed_webhooks ENABLE ROW LEVEL SECURITY;

-- Create policies (optional - for extra security)
-- Only service role can access all data
CREATE POLICY "Service role can access all" ON users
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can access all" ON oauth_identities
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can access all" ON usage_events
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can access all" ON subscriptions
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can access all" ON processed_webhooks
    FOR ALL USING (auth.role() = 'service_role');
