import os
from com.abnamro.llm.api.predict_service import load_model_and_tokenizer, predict_expense_type
from com.abnamro.llm.constants import Constants
from com.abnamro.llm.service.AccountOverviewService import AccountOverviewService
from flask import Flask, request, jsonify

# Initialize the Flask app for prediction service
app = Flask(__name__)

count = 0


# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # AccountNumber	Date	D-C-ind	Amount	Payment Method	Merchant	Description Lines	ExpenseType	ExpenseCategory

    if 'description' not in data:
        return jsonify({"error": "No description provided"}), 400
    if 'merchant' not in data:
        return jsonify({"error": "No merchant info provided"}), 400

    account_number = data['account_number']
    transaction_date = data['transaction_date']
    indicator = data['indicator']
    amount = data['amount']
    payment_method = data['payment_method']
    merchant = data['merchant']
    description_lines = data['description']
    # expense_type = data['description']
    # expense_category = data['description']

    try:
        # Load the model and tokenizer
        model_path = os.path.join("api/trained_model", "bert_expense_classification_model")
        print("model path : " + model_path)
        label_encoder_path = os.path.join("api/trained_model", "expense_label_encoder.pth")
        print("label encoder path : " + label_encoder_path)

        model, tokenizer, label_encoder = load_model_and_tokenizer(model_path, label_encoder_path)

        # Predict the ExpenseType using the model
        predicted_expense_type = predict_expense_type(merchant, model, tokenizer, label_encoder)

        expense_category = Constants.EXPENSE_TYPE_CATEGORY_DICT[predicted_expense_type]

        print("predicted expense type:", {predicted_expense_type}, "expense_category: ", {expense_category})

        return jsonify({
            "account_number": account_number,
            "transaction_date": transaction_date,
            "indicator": indicator,
            "amount": amount,
            "payment_method": payment_method,
            "merchant": merchant,
            "description_lines": description_lines,
            "expense_type": predicted_expense_type,
            "expense_category": expense_category
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def __saveAccountOverView(account_number, transaction_date, indicator, amount, payment_method, merchant,
                          description_lines, expense_type, expense_category, ):
    db_config = {
        'host': Constants.HOST,
        'port': Constants.PORT,
        'dbname': Constants.DATABASE_NAME,
        'user': Constants.USER_NAME,
        'password': Constants.PASSWORD
    }
    # Initialize the account overview service
    account_service = AccountOverviewService(db_config)
    # Connect to the database
    account_service.connect()
    # Path to the CSV file
    # csv_file_path = Constants.CSV_FILE_PATH
    # Insert records from the CSV file into the account_overview table
    account_service.insert_record(account_number, transaction_date, indicator, amount, payment_method, merchant,
                                  description_lines, expense_type, expense_category)
    # Close the connection to the database
    account_service.close_connection()


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.2', port=5002)
