-- 1️⃣ Clean up existing objects

DROP FUNCTION IF EXISTS public.match_documents_langchain(vector, integer);
DROP TABLE IF EXISTS public.documents;

-- 2️⃣ Install vector extension

DROP EXTENSION IF EXISTS vector;
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- 3️⃣ Create documents table

CREATE TABLE public.documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT,
  metadata JSONB,
  embedding extensions.vector(768)
);

-- 4️⃣ Enable Row Level Security (RLS)

ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

-- 5️⃣ Add open access policy

CREATE POLICY "Allow all access"
ON public.documents
FOR ALL
USING (true)
WITH CHECK (true);

-- 6️⃣ Create vector search function

CREATE OR REPLACE FUNCTION public.match_documents_langchain(
  query_embedding extensions.vector(768),
  match_count INT DEFAULT 5
)
RETURNS TABLE (
  id UUID,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
  SELECT
    id,
    content,
    metadata,
    1 - (embedding <#> query_embedding) AS similarity
  FROM
    public.documents
  ORDER BY
    embedding <#> query_embedding
  LIMIT match_count;