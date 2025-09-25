-- Create raw sales table - Idris
CREATE TABLE IF NOT EXISTS raw.sales_raw (
  "year" TEXT,
  "release_date" TEXT,
  "title" TEXT,
  "genre" TEXT,
  "international_box_office" TEXT,
  "domestic_box_office" TEXT,
  "worldwide_box_office" TEXT,
  "production_budget" TEXT,
  "opening_weekend" TEXT,
  "theatre_count" TEXT,
  "avg_run_per_theatre" TEXT,
  "runtime" TEXT,
  "keywords" TEXT,
  "creative_type" TEXT,
  "url" TEXT,
  "date_normalized" TEXT
);

-- import:
--command " "\\copy \"raw\".sales_raw (year, release_date, title, genre, international_box_office, domestic_box_office, worldwide_box_office, production_budget, opening_weekend, theatre_count, avg_run_per_theatre, runtime, keywords, creative_type, url, date_normalized) FROM '/Users/tristanriethorst/Library/CloudStorage/OneDrive-Personal/HvA/Database design/clean/final_sales_clean_v1_fixed.csv' DELIMITER ',' CSV ENCODING 'UTF8' QUOTE '\"' ESCAPE '''';""

-- Create raw metaclean table - Tristan
CREATE TABLE IF NOT EXISTS raw.metaclean_raw (
  url TEXT,
  title TEXT,
  studio TEXT,
  rating TEXT,
  runtime TEXT,
  "cast" TEXT,
  director TEXT,
  genre TEXT,
  summary TEXT,
  awards TEXT,
  metascore TEXT,
  userscore TEXT,
  reldate TEXT
);

-- import:
--command " "\\copy \"raw\".metaclean_raw (url, title, studio, rating, runtime, \"cast\", director, genre, summary, awards, metascore, userscore, reldate) FROM '/Users/tristanriethorst/Library/CloudStorage/OneDrive-Personal/HvA/Database design/clean/metaClean43Brightspace_cleanv4.csv' DELIMITER ',' CSV HEADER ENCODING 'UTF8' QUOTE '\"';""

-- Create raw userreviews table - Fatemeh
CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.userreviews_raw (
  url      TEXT,
  idvscore TEXT,
  reviewer TEXT,
  "dateP"  TEXT
);

-- import:
--command " "\\copy \"raw\".userreviews_raw (url, idvscore, reviewer, \"dateP\") FROM '/Users/tristanriethorst/Library/CloudStorage/OneDrive-Personal/HvA/Database design/clean/final_UserReviews_cleanv3.csv' DELIMITER ',' CSV ENCODING 'UTF8' QUOTE '\"' ESCAPE '''';""


-- Create raw expertreviews table - Sreejoni
CREATE TABLE IF NOT EXISTS raw.expertreviews_raw (
  "url" TEXT,
  "reviewer" TEXT,
  "dateP" TEXT,
  "idvscore" TEXT,
  "reviewer_normalized" TEXT,
  "date_normalized" TEXT
);

-- import:
--command " "\\copy \"raw\".expertreviews_raw (url, reviewer, \"dateP\", idvscore, reviewer_normalized, date_normalized) FROM '/Users/tristanriethorst/Library/CloudStorage/OneDrive-Personal/HvA/Database design/clean/final_ExpertReviewsCleaned_DB_clean.csv' DELIMITER ',' CSV HEADER ENCODING 'UTF8' QUOTE '\"';""

-- Create entity MOVIES - All
SET datestyle = 'ISO, DMY';

CREATE TABLE IF NOT EXISTS public.movie (
  url     TEXT PRIMARY KEY,
  title   TEXT,
  reldate DATE
);

INSERT INTO public.movie (url, title, reldate)
SELECT 
  "url",
  "title",
  NULLIF("reldate", '')::DATE
FROM raw.metaclean_raw;

-- Create entity PERFORMANCE - Idris
-- 1) Ensure movie has a unique key on (title, reldate)
ALTER TABLE public.movie
ADD CONSTRAINT movie_title_reldate_uk UNIQUE (title, reldate);

-- 2) Create PERFORMANCE with composite PK and FK to MOVIE(title,reldate)
CREATE TABLE IF NOT EXISTS public.performance (
  title                     TEXT,
  reldate                   DATE,
  production_budget         NUMERIC(14,2),
  worldwide_box_office      NUMERIC(14,2),
  opening_weekend_revenue   NUMERIC(14,2),
  CONSTRAINT performance_pk PRIMARY KEY (title, reldate),
  CONSTRAINT performance_movie_fk
    FOREIGN KEY (title, reldate)
    REFERENCES public.movie (title, reldate)
    ON UPDATE CASCADE
    ON DELETE RESTRICT
);

-- 3) Load from raw.sales_raw (using date_normalized = 'DD/MM/YYYY')
INSERT INTO public.performance (
  title, reldate, production_budget, worldwide_box_office, opening_weekend_revenue
)
SELECT DISTINCT ON (s."title", to_date(s."date_normalized",'DD/MM/YYYY'))
  s."title",
  to_date(s."date_normalized",'DD/MM/YYYY') AS reldate,
  NULLIF(regexp_replace(s."production_budget",     '[^0-9\.-]', '', 'g'), '')::NUMERIC(14,2),
  NULLIF(regexp_replace(s."worldwide_box_office",  '[^0-9\.-]', '', 'g'), '')::NUMERIC(14,2),
  NULLIF(regexp_replace(s."opening_weekend",       '[^0-9\.-]', '', 'g'), '')::NUMERIC(14,2)
FROM raw.sales_raw s
WHERE s."date_normalized" ~ '^\d{2}/\d{2}/\d{4}$'
  AND EXISTS (
    SELECT 1
    FROM public.movie m
    WHERE m.title = s."title"
      AND m.reldate = to_date(s."date_normalized",'DD/MM/YYYY')
  )
ORDER BY
  s."title",
  to_date(s."date_normalized",'DD/MM/YYYY');


-- Create entity DISTRIBUTION - Tristan 
-- 1) Create entity
CREATE TABLE IF NOT EXISTS public.distribution (
  title                   TEXT,
  reldate                 DATE,
  opening_weekend_revenue NUMERIC(14,2),
  theatre_count           INTEGER,
  CONSTRAINT distribution_pk PRIMARY KEY (title, reldate),
  CONSTRAINT distribution_movie_fk
    FOREIGN KEY (title, reldate)
    REFERENCES public.movie (title, reldate)
    ON UPDATE CASCADE
    ON DELETE RESTRICT
);

-- For the record we deleted some NULL rows:
SELECT *
FROM raw.sales_raw
WHERE date_normalized IS NULL;

DELETE FROM raw.sales_raw
WHERE date_normalized IS NULL;

-- 2) Load from raw.sales_raw 
SET datestyle = 'ISO, DMY';

INSERT INTO public.distribution (
  title, reldate, opening_weekend_revenue, theatre_count
)
SELECT DISTINCT ON (s."title", s."date_normalized"::date)
  s."title",
  s."date_normalized"::date AS reldate,
  CASE
    WHEN upper(btrim(s."opening_weekend")) IN ('', 'NULL') THEN NULL
    ELSE regexp_replace(s."opening_weekend", '[^0-9\.-]', '', 'g')::NUMERIC(14,2)
  END AS opening_weekend_revenue,

  CASE
    WHEN upper(btrim(s."theatre_count")) IN ('', 'NULL') THEN NULL
    ELSE regexp_replace(s."theatre_count", '[^0-9\.-]', '', 'g')::NUMERIC(14,2)
  END AS theatre_count

FROM raw.sales_raw s
WHERE s."date_normalized" IS NOT NULL
  AND EXISTS (
    SELECT 1
    FROM public.movie m
    WHERE m.title = s."title"
      AND m.reldate = s."date_normalized"::date
  )
ORDER BY s."title", s."date_normalized"::date;


-- Create entity USER_REVIEWS - Fatemeh
-- 1) Create entity with composite PK (url, reviewer) and FK to movie(url)
CREATE TABLE IF NOT EXISTS public.user_reviews (
  url      TEXT NOT NULL,
  reviewer TEXT NOT NULL,
  datep    DATE,
  idvscore INTEGER,
  CONSTRAINT user_reviews_pk PRIMARY KEY (url, reviewer),
  CONSTRAINT user_reviews_movie_fk
    FOREIGN KEY (url) REFERENCES public.movie(url)
      ON UPDATE CASCADE ON DELETE RESTRICT
);

-- Discovered that (url,reviewer) is not unique in raw.userreviews_raw
-- 1.1) Change PK to (url, reviewer, datep)
ALTER TABLE public.user_reviews
  DROP CONSTRAINT IF EXISTS user_reviews_pk;

ALTER TABLE public.user_reviews
  ADD CONSTRAINT user_reviews_pk
  PRIMARY KEY (url, reviewer, datep);

SET datestyle = 'ISO, DMY';

-- Veilige load uit raw.userreviews_raw
WITH src AS (
  SELECT
    r."url",
    r."reviewer",
    CASE
      WHEN r."dateP" IS NULL THEN NULL
      WHEN upper(btrim(r."dateP")) IN ('', 'NULL', '\N') THEN NULL
      WHEN btrim(r."dateP") ~ '^\d{1,2}/\d{1,2}/\d{4}$'
        THEN to_date(btrim(r."dateP"), 'DD/MM/YYYY')
      ELSE NULL
    END AS datep,
    NULLIF(btrim(r."idvscore"), '')::int AS idvscore
  FROM raw.userreviews_raw r
)
INSERT INTO public.user_reviews (url, reviewer, datep, idvscore)
SELECT
  s.url,
  s.reviewer,
  s.datep,
  s.idvscore
FROM src s
WHERE s.url IS NOT NULL
  AND s.reviewer IS NOT NULL
  AND s.datep IS NOT NULL              -- datep mag niet NULL zijn vanwege PK
  AND EXISTS (
    SELECT 1
    FROM public.movie m
    WHERE m.url = s.url
  )
ON CONFLICT (url, reviewer, datep) DO NOTHING;


-- Create entity EXPERT_REVIEWS - Sreejoni
-- 1) Create entity with composite PK (url, reviewer_normalized, date) and FK to movie(url)
CREATE TABLE IF NOT EXISTS public.expert_reviews (
  url                 TEXT NOT NULL,
  reviewer            TEXT,
  reviewer_normalized TEXT NOT NULL,
  "date"              DATE NOT NULL,
  idvscore            INTEGER,
  CONSTRAINT expert_reviews_pk PRIMARY KEY (url, reviewer_normalized, "date"),
  CONSTRAINT expert_reviews_movie_fk
    FOREIGN KEY (url) REFERENCES public.movie(url)
      ON UPDATE CASCADE ON DELETE RESTRICT
);

-- 2) Load from raw.expertreviews_raw (assumes date_normalized = 'DD/MM/YYYY')
SET datestyle = 'ISO, DMY';

INSERT INTO public.expert_reviews (url, reviewer, reviewer_normalized, "date", idvscore)
SELECT
  "url",
  "reviewer",
  "reviewer_normalized",
  NULLIF(btrim("date_normalized"), '')::date AS "date",
  NULLIF(btrim("idvscore"), '')::int
FROM raw.expertreviews_raw
WHERE "url" IS NOT NULL
  AND "reviewer_normalized" IS NOT NULL
  AND NULLIF(btrim("date_normalized"), '') IS NOT NULL
  AND EXISTS (
    SELECT 1
    FROM public.movie m
    WHERE m.url = raw.expertreviews_raw."url"
  )
ON CONFLICT (url, reviewer_normalized, "date") DO NOTHING;