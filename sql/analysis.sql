SELECT
    ROUND(100.0 * SUM(CASE WHEN card_payments > 5 THEN 1 ELSE 0 END) / COUNT(DISTINCT user_id), 2) AS card_adoption_pct,
    ROUND(100.0 * SUM(CASE WHEN bank_transfers > 2 THEN 1 ELSE 0 END) / COUNT(DISTINCT user_id), 2) AS bank_adoption_pct,
    ROUND(100.0 * SUM(CASE WHEN savings_vaults > 1 THEN 1 ELSE 0 END) / COUNT(DISTINCT user_id), 2) AS savings_adoption_pct,
    ROUND(100.0 * SUM(CASE WHEN crypto_usage > 1 THEN 1 ELSE 0 END) / COUNT(DISTINCT user_id), 2) AS crypto_adoption_pct,
    ROUND(100.0 * SUM(CASE WHEN stock_trading > 1 THEN 1 ELSE 0 END) / COUNT(DISTINCT user_id), 2) AS stock_adoption_pct,
    ROUND(100.0 * SUM(CASE WHEN intl_transfers > 1 THEN 1 ELSE 0 END) / COUNT(DISTINCT user_id), 2) AS intl_adoption_pct
FROM revolut_usage_50k_users_v2;


SELECT
    premium,
    ROUND(AVG(card_payments), 2)   AS avg_card_payments,
    ROUND(AVG(bank_transfers), 2)  AS avg_bank_transfers,
    ROUND(AVG(savings_vaults), 2)  AS avg_savings_vaults,
    ROUND(AVG(crypto_usage), 2)    AS avg_crypto_usage,
    ROUND(AVG(stock_trading), 2)   AS avg_stock_trading,
    ROUND(AVG(intl_transfers), 2)  AS avg_intl_transfers,
    COUNT(DISTINCT user_id)        AS total_users
FROM revolut_usage_50k_users_v2
GROUP BY premium;


SELECT
    month,
    ROUND(AVG(card_payments), 2)   AS avg_card_payments,
    ROUND(AVG(bank_transfers), 2)  AS avg_bank_transfers,
    ROUND(AVG(savings_vaults), 2)  AS avg_savings_vaults,
    ROUND(AVG(crypto_usage), 2)    AS avg_crypto_usage,
    ROUND(AVG(stock_trading), 2)   AS avg_stock_trading,
    ROUND(AVG(intl_transfers), 2)  AS avg_intl_transfers
FROM revolut_usage_50k_users_v2
GROUP BY month
ORDER BY month;



SELECT
    country,
    ROUND(AVG(card_payments), 2)   AS avg_card_payments,
    ROUND(AVG(bank_transfers), 2)  AS avg_bank_transfers,
    ROUND(AVG(savings_vaults), 2)  AS avg_savings_vaults,
    ROUND(AVG(crypto_usage), 2)    AS avg_crypto_usage,
    ROUND(AVG(stock_trading), 2)   AS avg_stock_trading,
    ROUND(AVG(intl_transfers), 2)  AS avg_intl_transfers,
    COUNT(DISTINCT user_id)        AS total_users
FROM revolut_usage_50k_users_v2
GROUP BY country
ORDER BY avg_crypto_usage DESC;



SELECT
    CASE
        WHEN age BETWEEN 18 AND 25 THEN '18-25'
        WHEN age BETWEEN 26 AND 35 THEN '26-35'
        WHEN age BETWEEN 36 AND 45 THEN '36-45'
        ELSE '46+'
    END AS age_group,
    ROUND(AVG(crypto_usage), 2)    AS avg_crypto,
    ROUND(AVG(savings_vaults), 2)  AS avg_savings,
    ROUND(AVG(card_payments), 2)   AS avg_card,
    ROUND(AVG(stock_trading), 2)   AS avg_stock,
    COUNT(DISTINCT user_id)        AS total_users
FROM revolut_usage_50k_users_v2
GROUP BY age_group
ORDER BY age_group;



SELECT
    CASE
        WHEN crypto_usage >= 10 THEN 'High Crypto'
        WHEN crypto_usage BETWEEN 5 AND 9 THEN 'Medium Crypto'
        ELSE 'Low Crypto'
    END AS crypto_segment,
    ROUND(AVG(savings_vaults), 2) AS avg_savings,
    ROUND(AVG(card_payments), 2)  AS avg_card_payments,
    COUNT(*)                      AS row_count
FROM revolut_usage_50k_users_v2
GROUP BY crypto_segment
ORDER BY avg_savings DESC;
