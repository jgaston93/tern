from app import pricing


def daily_report(orders):
    totals = [pricing.calc_total(o) for o in orders]
    return {"orders": len(orders), "revenue": sum(totals)}
