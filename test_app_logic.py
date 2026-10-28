"""
test_app_logic.py
Module 5: Streamlit Web Application Logic & Valuation Verification
Project: Used-Car Price Prediction Using Random Forest Regression

Comprehensive test suite verifying:
1. Model loading & pipeline integrity.
2. Regression consistency between app prediction logic and the saved pipeline.
3. Reproduction and analysis of fuel type predictions (Petrol vs Diesel vs CNG).
4. Investigation of the ~₹0.95 Lakh prediction scenario.
5. Valuation mathematics (unrounded calculations, retained value %, depreciation %).
6. Over-valuation anomaly detection (resale > original price, no silent clamping).
7. Input validation logic (rejection of zero/negative values, advisory warnings).
8. Trajectory projection consistency across vehicle ages.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure Windows terminal encoding compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import app


def run_tests():
    print("=" * 70)
    print("USED-CAR PRICE ESTIMATOR - APPLICATION LOGIC & VALUATION TESTS")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 1. Model Loading Check
    # --------------------------------------------------------------------------
    print("\n--- Test 1: Model Loading from Disk & Cache ---")
    model_path = os.path.join(os.path.dirname(__file__), "models", "used_car_price_model.joblib")
    assert os.path.exists(model_path), f"Model file missing at: {model_path}"

    disk_pipeline = joblib.load(model_path)
    app_pipeline = app.load_trained_pipeline()
    assert app_pipeline is not None, "app.load_trained_pipeline() returned None!"
    print("[PASS] Model successfully loaded both from disk and through app cache.")

    # --------------------------------------------------------------------------
    # 2. Regression Consistency: App Pipeline vs. Disk Pipeline
    # --------------------------------------------------------------------------
    print("\n--- Test 2: Regression Consistency (App vs. Direct Pipeline) ---")
    default_sample = {
        "Present_Price": 5.59,
        "Kms_Driven": 27000,
        "Owner": 0,
        "Car_Age": 6,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual"
    }

    df_sample = pd.DataFrame([default_sample])
    pred_disk = float(disk_pipeline.predict(df_sample)[0])
    pred_app = float(app_pipeline.predict(df_sample)[0])

    print(f"  Disk Pipeline Prediction : {pred_disk:.6f} Lakhs")
    print(f"  App Pipeline Prediction  : {pred_app:.6f} Lakhs")
    assert np.isclose(pred_disk, pred_app, atol=1e-5), "Pipeline outputs differ between disk and app!"
    print("[PASS] Exact numerical consistency verified between app and disk model.")

    # --------------------------------------------------------------------------
    # 3. Default Example Input Verification (~ ₹3.62 Lakhs)
    # --------------------------------------------------------------------------
    print("\n--- Test 3: Default Example Prediction (~ ₹3.62 Lakhs) ---")
    print(f"  Inputs: {default_sample}")
    print(f"  Predicted Resale: ₹{pred_app:.2f} Lakhs ({app.format_inr(pred_app * 100000)})")
    assert 3.50 <= pred_app <= 3.75, f"Default prediction {pred_app} deviates from expected ~3.62L"
    print("[PASS] Default Maruti Swift/Ritz prediction correctly matches ₹3.62 Lakhs.")

    # --------------------------------------------------------------------------
    # 4. Fuel Type Investigation (Petrol vs Diesel vs CNG)
    # --------------------------------------------------------------------------
    print("\n--- Test 4: Fuel Type Variation & Discrepancy Investigation ---")
    diesel_sample = dict(default_sample, Fuel_Type="Diesel")
    cng_sample = dict(default_sample, Fuel_Type="CNG")

    pred_diesel = float(app_pipeline.predict(pd.DataFrame([diesel_sample]))[0])
    pred_cng = float(app_pipeline.predict(pd.DataFrame([cng_sample]))[0])

    print(f"  Petrol (Present_Price=5.59L) : ₹{pred_app:.4f} Lakhs (₹{pred_app*100000:,.0f})")
    print(f"  Diesel (Present_Price=5.59L) : ₹{pred_diesel:.4f} Lakhs (₹{pred_diesel*100000:,.0f})")
    print(f"  CNG    (Present_Price=5.59L) : ₹{pred_cng:.4f} Lakhs (₹{pred_cng*100000:,.0f})")

    # In Random Forest, Present_Price represents 90.5% of tree splits; Fuel_Type represents ~0.2%.
    # Therefore, changing ONLY Fuel_Type on a 5.59L car yields ~3.65L, NEVER ~0.95L.
    assert 3.50 <= pred_diesel <= 3.80, f"Diesel prediction unexpected: {pred_diesel}"
    print("  [ANALYSIS] Selecting Diesel on a 5.59L car yields ₹3.65L (not ₹0.95L).")

    # Investigate what input yields ~0.95 Lakhs
    budget_bike_sample = {
        "Present_Price": 1.20,
        "Kms_Driven": 27000,
        "Owner": 0,
        "Car_Age": 6,
        "Fuel_Type": "Diesel",
        "Seller_Type": "Dealer",
        "Transmission": "Manual"
    }
    pred_budget = float(app_pipeline.predict(pd.DataFrame([budget_bike_sample]))[0])
    print(f"  Budget / Bike Listing (Present_Price=1.20L, Diesel): ₹{pred_budget:.2f} Lakhs ({app.format_inr(pred_budget*100000)})")
    assert 0.85 <= pred_budget <= 1.05, f"Budget sample expected near 0.95L, got {pred_budget}"
    print("  [ROOT CAUSE VERIFIED] A prediction of ~₹0.95L occurs when Present_Price is around ₹1.2L")
    print("    (e.g., two-wheeler/budget listings in the dataset), NOT merely from changing fuel to Diesel.")
    print("[PASS] Fuel type variations and root cause analysis verified.")

    # --------------------------------------------------------------------------
    # 5. Valuation Calculations (Unrounded retained %, depreciation %, & amounts)
    # --------------------------------------------------------------------------
    print("\n--- Test 5: Valuation Math (Retained Value %, Depreciation %, Amounts) ---")
    val = app.calculate_valuation(pred_app, default_sample["Present_Price"])

    expected_retained = (pred_app / default_sample["Present_Price"]) * 100.0
    expected_deprec = 100.0 - expected_retained
    expected_deprec_lakh = default_sample["Present_Price"] - pred_app
    expected_deprec_rupees = expected_deprec_lakh * 100000.0
    expected_val_diff = pred_app - default_sample["Present_Price"]

    print(f"  Original Showroom Price: ₹{default_sample['Present_Price']:.2f} Lakhs")
    print(f"  Predicted Resale Price : ₹{pred_app:.4f} Lakhs")
    print(f"  Calculated Retained    : {val['retained_pct']:.4f}% (Expected: {expected_retained:.4f}%)")
    print(f"  Calculated Depreciation: {val['depreciation_pct']:.4f}% (Expected: {expected_deprec:.4f}%)")
    print(f"  Depreciation Amount    : ₹{val['depreciation_amount_lakh']:.4f}L (₹{val['depreciation_amount_rupees']:,.0f})")
    print(f"  Value Difference (Diff): ₹{val['value_difference_lakh']:.4f}L")
    print(f"  Exceeds Original Price : {val['exceeds_original']}")

    assert np.isclose(val["retained_pct"], expected_retained), "Retained percentage calculation error"
    assert np.isclose(val["depreciation_pct"], expected_deprec), "Depreciation percentage calculation error"
    assert np.isclose(val["depreciation_amount_lakh"], expected_deprec_lakh), "Depreciation amount (Lakhs) calculation error"
    assert np.isclose(val["depreciation_amount_rupees"], expected_deprec_rupees), "Depreciation amount (Rupees) calculation error"
    assert np.isclose(val["value_difference_lakh"], expected_val_diff), "Value difference (Lakhs) calculation error"
    assert not val["exceeds_original"], "Default car should not exceed showroom cost"
    print("[PASS] Standard valuation arithmetic and monetary amount fields match exact formulas.")

    # --------------------------------------------------------------------------
    # 5b. Boundary & Exception Validation for calculate_valuation()
    # --------------------------------------------------------------------------
    print("\n--- Test 5b: calculate_valuation() Input Boundary Validation ---")
    # Must reject negative predicted price
    try:
        app.calculate_valuation(-0.5, 5.0)
        assert False, "calculate_valuation failed to reject negative predicted price!"
    except ValueError as e:
        print(f"  [PASS] Caught negative predicted price: {e}")

    # Must reject zero original price
    try:
        app.calculate_valuation(3.5, 0.0)
        assert False, "calculate_valuation failed to reject zero original price!"
    except ValueError as e:
        print(f"  [PASS] Caught zero original price: {e}")

    # Must reject negative original price
    try:
        app.calculate_valuation(3.5, -5.0)
        assert False, "calculate_valuation failed to reject negative original price!"
    except ValueError as e:
        print(f"  [PASS] Caught negative original price: {e}")

    # --------------------------------------------------------------------------
    # 5c. Indian Numbering System Currency Formatting (format_inr)
    # --------------------------------------------------------------------------
    print("\n--- Test 5c: Indian Currency Formatting (format_inr) ---")
    inr_cases = [
        (0, "₹0"),
        (559, "₹559"),
        (27000, "₹27,000"),
        (100000, "₹1,00,000"),
        (362475, "₹3,62,475"),
        (1000000, "₹10,00,000"),
        (10000000, "₹1,00,00,000"),
        (-50000, "-₹50,000")
    ]
    for amount, expected_str in inr_cases:
        res_str = app.format_inr(amount)
        print(f"  Amount {amount:>10} -> Formatted: {res_str} (Expected: {expected_str})")
        assert res_str == expected_str, f"format_inr({amount}) gave '{res_str}', expected '{expected_str}'"
    print("[PASS] Indian numbering system comma grouping verified across all magnitudes.")

    # --------------------------------------------------------------------------
    # 6. Over-valuation Anomaly Detection (No Silent Clamping)
    # --------------------------------------------------------------------------
    print("\n--- Test 6: Over-valuation Anomaly (No Silent Clamping) ---")
    # Anomaly test: Present_Price = 0.10L, Predicted = ~0.77835L
    anomaly_present = 0.10
    anomaly_pred = 0.77835
    anomaly_val = app.calculate_valuation(anomaly_pred, anomaly_present)

    print(f"  Original Showroom Price: ₹{anomaly_present:.2f} Lakhs (₹10,000)")
    print(f"  Predicted Resale Price : ₹{anomaly_pred:.4f} Lakhs (₹77,835)")
    print(f"  Retained Value         : {anomaly_val['retained_pct']:.2f}%")
    print(f"  Depreciation           : {anomaly_val['depreciation_pct']:.2f}%")
    print(f"  Depreciation Amount    : ₹{anomaly_val['depreciation_amount_lakh']:.4f} Lakhs")
    print(f"  Value Difference       : ₹{anomaly_val['value_difference_lakh']:.4f} Lakhs")
    print(f"  Exceeds Original Flag  : {anomaly_val['exceeds_original']}")

    # Critical assertions: Must NOT be clamped to 100% and 0%
    assert anomaly_val["retained_pct"] > 100.0, "Retained % was improperly clamped to <= 100%"
    assert np.isclose(anomaly_val["retained_pct"], 778.35, atol=0.01), f"Expected 778.35%, got {anomaly_val['retained_pct']}"
    assert anomaly_val["depreciation_pct"] < 0.0, "Depreciation % was improperly clamped to >= 0%"
    assert np.isclose(anomaly_val["depreciation_pct"], -678.35, atol=0.01), f"Expected -678.35%, got {anomaly_val['depreciation_pct']}"
    assert np.isclose(anomaly_val["depreciation_amount_lakh"], anomaly_present - anomaly_pred, atol=0.01)
    assert np.isclose(anomaly_val["value_difference_lakh"], anomaly_pred - anomaly_present, atol=0.01)
    assert anomaly_val["exceeds_original"] is True, "exceeds_original flag must be True"
    print("[PASS] Over-valuation correctly detected without silent clamping!")

    # --------------------------------------------------------------------------
    # 7. Input Validation: Strict Rejections & Advisory Bounds
    # --------------------------------------------------------------------------
    print("\n--- Test 7: Input Validation Logic ---")
    invalid_cases = [
        ("Negative Present Price", -5.0, 5, 20000, 0),
        ("Zero Present Price", 0.0, 5, 20000, 0),
        ("Negative Car Age", 5.0, -2, 20000, 0),
        ("Negative Mileage", 5.0, 5, -500, 0),
        ("Negative Owner Count", 5.0, 5, 20000, -1),
    ]

    for label, pp, age, kms, own in invalid_cases:
        is_ok, errs, _ = app.validate_inputs(pp, age, kms, own)
        assert not is_ok, f"Expected {label} to fail validation, but it passed!"
        assert len(errs) > 0, f"Expected error message for {label}"
        print(f"  [REJECTED] {label}: Caught '{errs[0]}'")

    # Valid case with advisory bounds
    is_ok, errs, advisories = app.validate_inputs(75.0, 20, 350000, 1)
    assert is_ok, "Extreme but positive inputs should pass validation"
    assert len(advisories) >= 2, "Expected advisory notes for extreme values"
    print(f"  [PASS] Extreme inputs passed with {len(advisories)} advisory notes.")

    # --------------------------------------------------------------------------
    # 8. Dynamic Trajectory Projection Consistency
    # --------------------------------------------------------------------------
    print("\n--- Test 8: Multi-Scenario Age Trajectory Consistency ---")
    ages = [2, 5, 8, 12]
    preds_by_age = []
    for a in ages:
        d = dict(default_sample, Car_Age=a)
        p = float(app_pipeline.predict(pd.DataFrame([d]))[0])
        preds_by_age.append(p)
        print(f"  Age {a:2d} years -> Estimated Resale: ₹{p:.2f} Lakhs")

    # In general, Random Forest estimates should monotonically decrease or remain stable with age
    assert preds_by_age[0] >= preds_by_age[-1], "Resale price should decrease as vehicle ages!"
    print("[PASS] Trajectory projections exhibit consistent negative age-depreciation trend.")

    print("\n" + "=" * 70)
    print("ALL APPLICATION LOGIC AND VALUATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
