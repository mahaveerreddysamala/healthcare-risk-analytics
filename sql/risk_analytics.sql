-- High-risk population by insurance
SELECT insurance_type,
       COUNT(*) AS patients,
       SUM(high_risk) AS high_risk_patients,
       ROUND(100.0 * AVG(high_risk), 2) AS high_risk_rate_pct
FROM patients
GROUP BY insurance_type
ORDER BY high_risk_rate_pct DESC;

-- Risk by smoking status
SELECT smoker,
       COUNT(*) AS patients,
       ROUND(AVG(bmi), 2) AS avg_bmi,
       ROUND(AVG(systolic_bp), 2) AS avg_systolic_bp,
       ROUND(AVG(high_risk), 3) AS risk_rate
FROM patients
GROUP BY smoker;
