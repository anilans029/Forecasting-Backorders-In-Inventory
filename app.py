from flask import Flask, request, render_template
from flask_cors import cross_origin
import pandas as pd
from backorder.ml.model.esitmator import BackorderData
from backorder.pipeline.prediciton_pipeline import PredictionPipeline

app = Flask(__name__)

prediction_pipeline = PredictionPipeline()


@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        national_inv= float(request.form['national_inv'])
        lead_time=float(request.form['lead_time'])
        in_transit_qty= float(request.form['in_transit_qty'])
        forecast_6_month= float(request.form['forecast_6_month'])
        sales_6_month = float(request.form['sales_6_month'])
        min_bank= float(request.form['min_bank'])
        potential_issue= request.form['potential_issue']
        pieces_past_due= float(request.form['pieces_past_due'])
        perf_6_month_avg= float(request.form['perf_6_month_avg'])
        local_bo_qty=float(request.form['local_bo_qty'])
        deck_risk=request.form['deck_risk']
        oe_constraint=request.form['oe_constraint']
        ppap_risk=request.form['ppap_risk']
        stop_auto_buy=request.form['stop_auto_buy']
        rev_stop=request.form['rev_stop']

        backorder_data=  BackorderData(national_inv= national_inv,
                                    lead_time=lead_time,
                                    in_transit_qty=in_transit_qty,
                                    forecast_6_month=forecast_6_month,
                                    sales_6_month=sales_6_month,
                                    min_bank=min_bank,
                                    potential_issue=potential_issue,
                                    pieces_past_due=pieces_past_due,
                                    perf_6_month_avg=perf_6_month_avg,
                                    local_bo_qty=local_bo_qty,
                                    deck_risk=deck_risk,
                                    oe_constraint=oe_constraint,
                                    ppap_risk=ppap_risk,
                                    stop_auto_buy=stop_auto_buy,
                                    rev_stop=rev_stop,
                                )
        input_df = backorder_data.get_backorder_input_data_frame()
        output = prediction_pipeline.start_single_instance_prediction(dataframe= input_df)

    return render_template('index.html',prediction_text=f'{output}')


    # return render_template("home1.html")
if __name__ == "__main__":
    app.run(debug=True, port= 8000)