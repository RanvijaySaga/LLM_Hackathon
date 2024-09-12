import dataclasses


class AccountOverview:
    accountNumber: str
    transactionDate: str
    indicator: str
    amount: float
    paymentMethod: str
    merchant: str
    descriptionLines: str
    expenseType: str

    def __init__(self, accountNumber, transactionDate, indicator, amount, paymentMethod, merchant, descriptionLines,
                 expenseType):
        self.accountNumber = accountNumber
        self.transactionDate = transactionDate
        self.indicator = indicator
        self.amount = amount
        self.paymentMethod = paymentMethod
        self.merchant = merchant
        self.descriptionLines = descriptionLines
        self.expenseType = expenseType
