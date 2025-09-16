-- Fill valid dates into the existing dateP_date column
UPDATE expert_reviews_small
SET dateP_date = TO_DATE(dateP, 'Mon DD YYYY')
WHERE dateP ~ '^[A-Z][a-z]{2} [0-9]{1,2} [0-9]{4}$';

-- Replace old text column with the new date column
ALTER TABLE expert_reviews_small
DROP COLUMN IF EXISTS dateP;

ALTER TABLE expert_reviews_small
RENAME COLUMN dateP_date TO dateP;

ALTER TABLE expert_reviews_small
ALTER COLUMN idvscore TYPE NUMERIC(5,2)
USING ROUND(idvscore::NUMERIC, 2);

SELECT idvscore, COUNT(*) 
FROM expert_reviews_small
GROUP BY idvscore
LIMIT 10;

ALTER TABLE expert_reviews_small
ADD COLUMN id SERIAL PRIMARY KEY;

SELECT COUNT(*)
FROM expert_reviews_small
WHERE idvscore IS NULL;

UPDATE expert_reviews_small
SET idvscore = 0
WHERE idvscore IS NULL;

ALTER TABLE expert_reviews_small
ALTER COLUMN idvscore SET NOT NULL;






