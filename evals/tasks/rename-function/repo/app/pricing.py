TAX_RATE = 0.08


def calc_total(items, discount=0.0):
    subtotal = sum(price * qty for price, qty in items)
    subtotal -= subtotal * discount
    return round(subtotal * (1 + TAX_RATE), 2)
