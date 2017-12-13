import QuantLib as ql
import pandas as pd
import math

def calibration_report(helpers, strikes):
    columns = ["Strikes", "Market Price", "Model Price",  "Relative Error"]
    report_data=[]
    cum_err = 0.0
    for i, option in enumerate(helpers):
        model_price = option.modelValue()
        market_price = option.marketValue()
        rel_error = (model_price / market_price - 1.0)
        report_data.append((strikes[i], model_price, market_price, rel_error))
        cum_err += abs(rel_error)
    cum_err = cum_err / len(heston_helpers)
    print("Cumulative error Price: %7.5f" % cum_err)
    return pd.DataFrame(report_data, columns=columns, index=['']*len(report_data))

def european_report(calculation_date, maturity_date, calendar, spot_price,
                    day_count, risk_free_rate, dividend_rate, strikes, black_var_surface):
    report_data=[]
    types = [ql.Option.Call, ql.Option.Put]
    columns = ["Strikes", "Type", "BSM Price", "Black Price"]
    ql.Settings.instance().evaluationDate = calculation_date
    exercise = ql.EuropeanExercise(maturity_date)
    spot_handle = ql.QuoteHandle(
        ql.SimpleQuote(spot_price)
    )
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count)
    )
    dividend_yield = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, dividend_rate, day_count)
    )
    expiry = (maturity_date - calculation_date)/365
    for type in types:
        for strike in strikes:
            # construct the European Option
            if type==1:
                type2 = "Call"
            else:
                type2 = "Put"

            payoff = ql.PlainVanillaPayoff(type, strike)
            european_option = ql.VanillaOption(payoff, exercise)
            vol = black_var_surface.blackVol(expiry, strike)
            flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(calculation_date, calendar, vol, day_count)
            )
            bsm_process = ql.BlackScholesMertonProcess(spot_handle,
                                                       dividend_yield,
                                                       flat_ts,
                                                       flat_vol_ts)
            european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
            bs_price = european_option.NPV()
            strike_payoff = ql.PlainVanillaPayoff(type, strike)
            T = flat_ts.dayCounter().yearFraction(calculation_date, maturity_date)
            stdev = vol*math.sqrt(T)
            discount = flat_ts.discount(maturity_date)
            black = ql.BlackCalculator(strike_payoff,
                                       spot_price,
                                       stdev,
                                       discount)
            report_data.append((strike, type2 , bs_price, black.value()))
    return pd.DataFrame(report_data, columns=columns, index=[''] * len(report_data))

# option data
maturity_date = ql.Date(1, 1, 2018)
spot_price = 2300
strike_price = spot_price
volatility = 0.20 # the historical vols for a year
dividend_rate = 0.0
option_type = ql.Option.Call
risk_free_rate = 0.01
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()
calculation_date = ql.Date(1, 1, 2017)
ql.Settings.instance().evaluationDate = calculation_date
tenor = ql.Period(ql.Monthly)
business_convention = ql.Following
termination_business_convention = ql.Following
date_generation = ql.DateGeneration.Forward
end_of_month = False
schedule = ql.Schedule(calculation_date,
                    maturity_date,
                    tenor,
                    calendar,
                    business_convention,
                    termination_business_convention,
                    date_generation,
                    end_of_month)
expiration_dates = list(schedule)
strike_ratios = [.6, .7, .8, .9, 1, 1.1, 1.2, 1.3,1.4]
strikes = [sr*spot_price for sr in strike_ratios]
vols = [[volatility for i in range(len(expiration_dates))] for j in range(len(strikes))]
black_var_surface = ql.BlackVarianceSurface(calculation_date, calendar,
                                            expiration_dates, strikes,
                                            vols, day_count)
option_report = european_report(calculation_date, maturity_date,calendar, spot_price,
                                day_count, risk_free_rate, dividend_rate, strikes, black_var_surface)
print(option_report)


# These are sample parameters, will be calibrated
spot_handle = ql.QuoteHandle(
    ql.SimpleQuote(spot_price)
)
flat_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, risk_free_rate, day_count)
)
dividend_yield = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, dividend_rate, day_count)
)
bates_params = {'v0': .05, 'kappa': 3.7, 'theta': .05,
                'sigma': 1.0, 'rho': -.6, 'lambda': .1, 'nu': -.5, 'delta': 0.3}
bp = ql.BatesProcess(flat_ts, dividend_yield, spot_handle, bates_params['v0'],
                    bates_params['kappa'], bates_params['theta'],
                    bates_params['sigma'], bates_params['rho'],
                    bates_params['lambda'], bates_params['nu'],
                    bates_params['delta'])
bm = ql.BatesModel(bp)
be = ql.BatesEngine(bm)


heston_helpers = []
black_var_surface.setInterpolation("bicubic")
date = expiration_dates[-1]
engine = ql.BatesEngine(bm)
for j, s in enumerate(strikes):
    t = (date - calculation_date)
    p = ql.Period(t, ql.Days)
    sigma = vols[j][-1]
    helper = ql.HestonModelHelper(p, calendar, spot_price, s,
                                ql.QuoteHandle(ql.SimpleQuote(sigma)),
                                flat_ts,
                                dividend_yield)
    helper.setPricingEngine(engine)
    heston_helpers.append(helper)
lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
bm.calibrate(heston_helpers, lm,
            ql.EndCriteria(50000, 500, 1.0e-8,1.0e-8, 1.0e-8))
print(bm.params())
print(len(heston_helpers))

cr = calibration_report(heston_helpers, strikes)
print(cr)
