export function normalizeCoverageLevel(s) {
  const t = (s || "").toLowerCase();

  if (t.includes("polni")) return "polni_kasko";
  if (t.includes("delni")) return "delni_kasko";
  if (t.includes("osnov")) return "osnovno";

  return null;
}

export function calculate_premium(vehicleAge, horsepower, city, coverageLevel) {
  const age = Number(vehicleAge);
  const hp = Number(horsepower);
  const lvl = String(coverageLevel || "");
  const cityStr = String(city || "Other");

  if (!Number.isFinite(age) || age < 0) throw new Error("vehicleAge must be a non-negative number");
  if (!Number.isFinite(hp) || hp <= 0) throw new Error("horsepower must be a positive number");
  if (!lvl) throw new Error("coverageLevel is required");

  let base = 220;

  const ageFactor = Math.max(0.78, Math.min(1.15, 1.10 - age * 0.02));
  const hpFactor = Math.max(0.85, Math.min(1.60, 0.85 + hp / 200));

  const cityKey = cityStr.toLowerCase();
  const cityFactorMap = {
    "ljubljana": 1.05,
    "maribor": 1.03,
    "koper": 1.02,
    "celje": 1.01,
    "kranj": 1.02
  };
  const cityFactor = cityFactorMap[cityKey] || 1.0;

  const coverageFactorMap = {
    "osnovno": 1.0,
    "delni_kasko": 1.28,
    "polni_kasko": 1.55
  };
  const coverageFactor = coverageFactorMap[lvl] ?? 1.0;

  const annual = Math.round(base * ageFactor * hpFactor * cityFactor * coverageFactor);
  const monthly = Math.round(annual / 12);

  return {
    annual_eur: annual,
    monthly_eur: monthly,
    breakdown: { base, ageFactor, hpFactor, cityFactor, coverageFactor }
  };
}
