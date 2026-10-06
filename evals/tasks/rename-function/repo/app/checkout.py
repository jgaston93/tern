from app.pricing import calc_total


def checkout(cart, member=False):
    discount = 0.1 if member else 0.0
    total = calc_total(cart, discount)
    return {"total": total, "items": len(cart)}
