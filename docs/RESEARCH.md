# Research foundation

The model is built from a small set of signals that each have published, out-of-sample evidence, and it
combines them with simple rules instead of a fitted black box. This document maps every module to the
research behind it, records what the evidence does not support, and sets the routine for turning new
papers into model changes.

## Five design principles

1. **Simple, robust signals over fitted ones.** Most equity-premium predictors that look good in-sample fail
   out of sample (Welch & Goyal 2008). Simple models also hold up against machine learning in yield-curve
   forecasting (Singh, Ogunfunmi & Das 2026).
2. **Combine many weak signals.** Averaging forecasts beats picking a single best predictor (Rapach, Strauss &
   Zhou 2010). The 11 flags and the 10-part risk score follow this.
3. **Persistence before action.** Monthly macro data are noisy, so inflation states need consecutive
   month-ends. Market crises are fast, so credit and volatility triggers act immediately.
4. **Every call is explainable.** Each view lists the readings that produced it and the thresholds that would
   change it.
5. **Assume overfitting until shown otherwise.** Many published factors are false discoveries (Harvey, Liu &
   Zhu 2016). Backtests carry a measurable probability of overfitting (Bailey et al. 2017). Thresholds change
   only through the review routine below.

## Module map

| Module | Question it answers | Code | Research basis | Known limits |
| --- | --- | --- | --- | --- |
| Recession risk | Is a downturn likely within 12 months? | `recession_prob_12m`, GROWTH_WEAK, RECESSION | Estrella & Hardouvelis 1991; Estrella & Mishkin 1998; Engstrom & Sharpe 2019; Sahm 2019; Moench & Stein 2026 | The 2022–24 inversion produced no recession; the 2024 Sahm reading was a false positive |
| Credit & financial conditions | Is stress spreading through funding markets? | CREDIT_STRESS, CREDIT_CRISIS, CREDIT_HEALING | Gilchrist & Zakrajšek 2012; López-Salido, Stein & Zakrajšek 2017; Adrian, Boyarchenko & Giannone 2019; Brave & Butters 2011 | FRED keeps only 3 years of ICE spreads, so the job archives its own copy |
| Inflation & policy | Is inflation forcing tighter policy? | INFL_HOT, INFL_COOL, HAWKISH, PIVOT | Taylor 1993; Bernanke & Kuttner 2005; Hamilton 1983; Kilian 2009; Benigno et al. 2022 | Supply-driven oil shocks hurt more than demand-driven rallies (Kilian 2009); the rule does not tell them apart yet |
| Regimes | Which state is the economy in, and when does it change? | `run_state_machine` | Hamilton 1989; Ang & Bekaert 2002; Kritzman, Page & Turkington 2012 | Rule-based, not estimated; transition dates must be checked against history |
| Short-term trend (1–3 months) | Is the price trend with or against you? | `short_term()` | Jegadeesh & Titman 1993; Moskowitz & Grinblatt 1999; Moskowitz, Ooi & Pedersen 2012; Faber 2007; Neely et al. 2014 | Momentum crashes in sharp rebounds (Daniel & Moskowitz 2016), so negative trend is discounted in Recovery |
| Medium-term fit (6–12 months) | Which sectors and countries suit this regime? | `medium_term()`, `regime_tilts` | Asness, Moskowitz & Pedersen 2013; Conover et al. 2008; Moench & Stein 2026 | Business-cycle sector rotation adds little even with perfect hindsight (Stangl, Jacobsen & Visaltanachoti 2009), so tilts carry only part of the score |
| Volatility | Is the market in a panic or a release? | VIX rules, Amber | Moreira & Muir 2017; Whaley 2000 | VIX starts in 1990; earlier history needs realised volatility |
| Long-term valuation (3–5+ years) | What return does today's price imply? | `long_term()` | Campbell & Shiller 1988, 1998; Fama & French 1989; the excess CAPE yield (CAPE earnings yield minus the real 10-year yield) | Predicts long-run returns, not timing; sector and country valuations need extra data |
| FOMC timing (not yet used) | Do returns cluster around Fed meetings? | — | Lucca & Moench 2015; Cieslak, Morse & Vissing-Jorgensen 2019 | Candidate for a future short-term overlay |

## Reading list

**Recession risk**
- Estrella, A. and Hardouvelis, G. (1991). The term structure as a predictor of real economic activity. *Journal of Finance* 46(2).
- Estrella, A. and Mishkin, F. (1998). Predicting U.S. recessions: financial variables as leading indicators. *Review of Economics and Statistics* 80(1).
- Engstrom, E. and Sharpe, S. (2019). The near-term forward yield spread as a leading indicator. *Financial Analysts Journal* 75(4).
- Sahm, C. (2019). Direct stimulus payments to individuals. In *Recession Ready*, Hamilton Project / Brookings.
- Moench, E. and Stein, T. (2026). Equity premium predictability over the business cycle. *Journal of Financial and Quantitative Analysis* 61(3): 1216–1246. Finds equity returns fall just before business-cycle peaks and recover during the recession, and that term-spread recession probabilities improve equity-premium forecasts out of sample. This is the empirical basis for the Recovery regime.
- Petrosky-Nadeau, N., Sung, Y. and Wilson, D. (2026). Do vibes predict recessions? Evidence from a big-data forecasting framework. FRBSF Working Paper 2026-14. Separates survey ("soft") signals from hard data.
- Banque de France (2026). Market-implied recession probability from machine learning on financial variables (PICON). *Bulletin* 264/2.

**Credit and financial conditions**
- Gilchrist, S. and Zakrajšek, E. (2012). Credit spreads and business cycle fluctuations. *American Economic Review* 102(4).
- López-Salido, D., Stein, J. and Zakrajšek, E. (2017). Credit-market sentiment and the business cycle. *Quarterly Journal of Economics* 132(3). Very tight spreads predict later widening, the basis for treating HY below 300 bp as complacency.
- Adrian, T., Boyarchenko, N. and Giannone, D. (2019). Vulnerable growth. *American Economic Review* 109(4).
- Brave, S. and Butters, R. A. (2011). Monitoring financial stability: a financial conditions index approach. Chicago Fed *Economic Perspectives* 35(1).

**Inflation, energy and policy**
- Taylor, J. (1993). Discretion versus policy rules in practice. *Carnegie-Rochester Conference Series on Public Policy* 39.
- Bernanke, B. and Kuttner, K. (2005). What explains the stock market's reaction to Federal Reserve policy? *Journal of Finance* 60(3).
- Hamilton, J. (1983). Oil and the macroeconomy since World War II. *Journal of Political Economy* 91(2).
- Kilian, L. (2009). Not all oil price shocks are alike. *American Economic Review* 99(3).
- Benigno, G., di Giovanni, J., Groen, J. and Noble, A. (2022). The GSCPI: a new barometer of global supply chain pressures. NY Fed Staff Report 1017.

**Regimes**
- Hamilton, J. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica* 57(2).
- Ang, A. and Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies* 15(4).
- Kritzman, M., Page, S. and Turkington, D. (2012). Regime shifts: implications for dynamic strategies. *Financial Analysts Journal* 68(3).

**Trend, momentum and rotation**
- Jegadeesh, N. and Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance* 48(1).
- Moskowitz, T. and Grinblatt, M. (1999). Do industries explain momentum? *Journal of Finance* 54(4).
- Moskowitz, T., Ooi, Y. H. and Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics* 104(2).
- Asness, C., Moskowitz, T. and Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance* 68(3).
- Daniel, K. and Moskowitz, T. (2016). Momentum crashes. *Journal of Financial Economics* 122(2).
- Faber, M. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management* 9(4).
- Neely, C., Rapach, D., Tu, J. and Zhou, G. (2014). Forecasting the equity risk premium: the role of technical indicators. *Management Science* 60(7).
- Conover, C. M., Jensen, G., Johnson, R. and Mercer, J. (2008). Sector rotation and monetary conditions. *Journal of Investing* 17(1).
- Stangl, J., Jacobsen, B. and Visaltanachoti, N. (2009). Sector rotation over business cycles. Working paper, Massey University.

**Volatility**
- Whaley, R. (2000). The investor fear gauge. *Journal of Portfolio Management* 26(3).
- Moreira, A. and Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance* 72(4).

**Valuation and expected returns**
- Campbell, J. and Shiller, R. (1988). Stock prices, earnings, and expected dividends. *Journal of Finance* 43(3).
- Campbell, J. and Shiller, R. (1998). Valuation ratios and the long-run stock market outlook. *Journal of Portfolio Management* 24(2).
- Fama, E. and French, K. (1989). Business conditions and expected returns on stocks and bonds. *Journal of Financial Economics* 25(1).

**Forecasting discipline**
- Welch, I. and Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies* 21(4).
- Campbell, J. and Thompson, S. (2008). Predicting excess stock returns out of sample: can anything beat the historical average? *Review of Financial Studies* 21(4).
- Rapach, D., Strauss, J. and Zhou, G. (2010). Out-of-sample equity premium prediction: combination forecasts and links to the real economy. *Review of Financial Studies* 23(2).
- Harvey, C., Liu, Y. and Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies* 29(1).
- Bailey, D., Borwein, J., López de Prado, M. and Zhu, Q. (2017). The probability of backtest overfitting. *Journal of Computational Finance* 20(4).
- Singh, A., Ogunfunmi, T. and Das, S. (2026). Yield curve forecasting using machine learning and econometrics: a comparative analysis. arXiv:2605.09842.

**FOMC timing (future overlay)**
- Lucca, D. and Moench, E. (2015). The pre-FOMC announcement drift. *Journal of Finance* 70(1).
- Cieslak, A., Morse, A. and Vissing-Jorgensen, A. (2019). Stock returns over the FOMC cycle. *Journal of Finance* 74(5).

## From research watch to model change

The dashboard's Research watch tab lists new papers from NBER, the Federal Reserve Board, Liberty Street
Economics and arXiv, scored against the modules above. It never changes the model by itself. Review it
monthly and treat a paper as a reason to change the model only if it passes all four gates:

1. **Out-of-sample evidence** across at least two business cycles, or a clearly separate test period.
2. **An economic mechanism**, not only a statistical fit.
3. **Free, timely data**: a signal you cannot refresh daily or monthly does not belong here.
4. **No damage to the six reference episodes** in the booklet (1979–82, 2000–02, 2007–09, 2020, 2022–23, 2026).

When a paper passes, change `config/thresholds.yaml`, `config/universe.yaml` or the code, and record the date,
the paper and the reason in `docs/MODEL_CHANGES.md`.
