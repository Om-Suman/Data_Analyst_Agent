import io
import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_config_endpoints(client):
    # Get config
    res = client.get("/api/config")
    assert res.status_code == 200
    cfg = res.json()
    assert "has_api_key" in cfg
    assert "primary_model" in cfg

    # Update config
    update_res = client.post(
        "/api/config",
        json={
            "hf_api_key": "hf_test_secret_key_12345",
            "primary_model": "Qwen/Qwen3-32B",
            "theme": "light",
            "temperature": 0.5,
        },
    )
    assert update_res.status_code == 200
    updated = update_res.json()
    assert updated["has_api_key"] is True
    assert updated["primary_model"] == "Qwen/Qwen3-32B"
    assert updated["theme"] == "light"
    assert updated["temperature"] == 0.5
    # Ensure raw API key is never exposed
    assert "hf_test_secret_key_12345" not in updated["api_key_masked"]
    assert updated["api_key_masked"].endswith("2345")


def test_dataset_sample_and_preview(client):
    # Load sample dataset
    res = client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["active_dataset"] == "Sales Data"

    # List datasets
    list_res = client.get("/api/datasets")
    assert list_res.status_code == 200
    datasets = list_res.json()["datasets"]
    assert len(datasets) == 1
    assert datasets[0]["name"] == "Sales Data"
    assert datasets[0]["rows"] == 500

    # Preview active dataset
    prev_res = client.get("/api/datasets/preview?limit=10")
    assert prev_res.status_code == 200
    preview = prev_res.json()
    assert preview["name"] == "Sales Data"
    assert len(preview["data"]) == 10
    assert "sales" in preview["numeric_cols"]


def test_file_upload(client):
    csv_content = b"id,name,amount,category\n1,Alice,100,A\n2,Bob,200,B\n3,Charlie,300,A\n"
    files = {"files": ("test_data.csv", csv_content, "text/csv")}
    res = client.post("/api/datasets/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["active_dataset"] == "test_data.csv"

    # Verify preview
    prev_res = client.get("/api/datasets/preview")
    assert prev_res.status_code == 200
    preview = prev_res.json()
    assert preview["rows"] == 3
    assert preview["cols"] == 4


def test_cleaning_and_rollback(client):
    # Load sales data
    client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})

    # Check quality overview
    q_res = client.get("/api/cleaning/quality")
    assert q_res.status_code == 200
    q_data = q_res.json()
    assert "quality_score" in q_data
    assert "quality_grade" in q_data

    # Preview cleaning
    prev_clean = client.post(
        "/api/cleaning/preview",
        json={
            "missing_strategy": "mean",
            "remove_duplicates": True,
            "fix_dtypes": True,
            "normalize_column_names": True,
            "outlier_method": "none",
        },
    )
    assert prev_clean.status_code == 200
    clean_report = prev_clean.json()
    assert "quality_score_after" in clean_report

    # Apply cleaning
    apply_res = client.post(
        "/api/cleaning/apply",
        json={
            "missing_strategy": "mean",
            "remove_duplicates": True,
            "fix_dtypes": True,
            "normalize_column_names": True,
            "outlier_method": "none",
        },
    )
    assert apply_res.status_code == 200

    # Check version history
    v_res = client.get("/api/cleaning/versions")
    assert v_res.status_code == 200
    v_data = v_res.json()
    assert len(v_data["versions"]) >= 2

    # Rollback to version 1
    rb_res = client.post("/api/cleaning/rollback", json={"version": 1})
    assert rb_res.status_code == 200


def test_explorer_endpoints(client):
    client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})

    # Browse table with pagination
    browse_res = client.post(
        "/api/explorer/browse",
        json={"page": 1, "page_size": 25, "sort_by": "sales", "sort_dir": "desc"},
    )
    assert browse_res.status_code == 200
    browse_data = browse_res.json()
    assert browse_data["total_rows"] == 500
    assert len(browse_data["data"]) == 25

    # Correlations
    corr_res = client.post(
        "/api/explorer/correlations",
        json={"columns": ["sales", "units", "profit"], "method": "pearson"},
    )
    assert corr_res.status_code == 200
    corr_data = corr_res.json()
    assert "matrix" in corr_data
    assert len(corr_data["top_pairs"]) > 0

    # Distribution
    dist_res = client.post(
        "/api/explorer/distribution",
        json={"column": "sales", "chart_type": "Histogram"},
    )
    assert dist_res.status_code == 200
    dist_data = dist_res.json()
    assert "figure_spec" in dist_data

    # Column profile
    prof_res = client.get("/api/explorer/profile/sales")
    assert prof_res.status_code == 200
    prof_data = prof_res.json()
    assert prof_data["is_numeric"] is True
    assert prof_data["numeric_stats"]["min"] is not None


def test_visualizations_endpoint(client):
    client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})

    # Bar chart
    bar_res = client.post(
        "/api/visualizations/generate",
        json={"chart_type": "Bar Chart", "x": "region", "y": "sales"},
    )
    assert bar_res.status_code == 200
    assert "figure_spec" in bar_res.json()

    # Scatter plot
    scatter_res = client.post(
        "/api/visualizations/generate",
        json={"chart_type": "Scatter Plot", "x": "sales", "y": "profit"},
    )
    assert scatter_res.status_code == 200
    assert "figure_spec" in scatter_res.json()


def test_forecasting_endpoint(client):
    client.post("/api/datasets/sample", json={"sample_name": "Finance Data"})

    res = client.post(
        "/api/forecasting/run",
        json={
            "target_col": "close",
            "method": "Moving Average",
            "horizon": 14,
            "window": 5,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["method"] == "Moving Average"
    assert len(data["forecast_values"]) == 14
    assert len(data["confidence_lower"]) == 14

    # Test download
    dl_res = client.get("/api/forecasting/download")
    assert dl_res.status_code == 200
    assert "Forecast" in dl_res.text


def test_anomaly_detection_endpoint(client):
    client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})

    res = client.post(
        "/api/anomalies/run",
        json={"method": "Z-Score", "threshold": 2.5},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["method"] == "Z-Score"
    assert "n_anomalies" in data
    assert "figure_spec" in data


def test_reports_endpoints(client):
    client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})

    # HTML report
    html_res = client.post(
        "/api/reports/html",
        json={"include_sample": True, "include_stats": True, "include_insights": False},
    )
    assert html_res.status_code == 200
    assert "<!DOCTYPE html>" in html_res.text

    # Excel report
    excel_res = client.get("/api/reports/excel")
    assert excel_res.status_code == 200
    assert len(excel_res.content) > 1000

    # Profile report
    prof_res = client.get("/api/reports/profile")
    assert prof_res.status_code == 200
    assert len(prof_res.json()["numeric_columns"]) > 0


def test_quick_insights(client):
    client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})
    res = client.get("/api/insights/quick")
    assert res.status_code == 200
    data = res.json()
    assert len(data["insights"]) > 0


def test_document_qa_endpoint(client):
    text_content = "Antigravity is an advanced agentic AI platform. It supports multi-agent systems and safe execution sandboxes."
    files = {"files": ("readme.txt", text_content.encode("utf-8"), "text/plain")}
    upload_res = client.post("/api/datasets/upload", files=files)
    assert upload_res.status_code == 200

    qa_res = client.post(
        "/api/document/qa",
        json={"question": "What is Antigravity?"},
    )
    assert qa_res.status_code == 200
    data = qa_res.json()
    assert "answer" in data
    assert len(data["sources"]) > 0


def test_sql_query_endpoint(client):
    client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})
    res = client.post(
        "/api/query/sql",
        json={"query": "SELECT region, SUM(sales) as total_sales FROM df GROUP BY region ORDER BY total_sales DESC"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "region" in data["columns"]
    assert "total_sales" in data["columns"]
    assert len(data["rows"]) > 0

    # Test forbidden statement
    bad_res = client.post(
        "/api/query/sql",
        json={"query": "DROP TABLE df"},
    )
    assert bad_res.status_code == 200
    bad_data = bad_res.json()
    assert bad_data["success"] is False
    assert "restriction" in bad_data["error"].lower()


def test_column_transform_endpoint(client):
    client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})

    # Rename
    res_rename = client.post(
        "/api/cleaning/transform-column",
        json={"column": "profit", "operation": "rename", "new_name": "net_profit"},
    )
    assert res_rename.status_code == 200
    assert "net_profit" in res_rename.json()["columns"]

    # String case
    res_case = client.post(
        "/api/cleaning/transform-column",
        json={"column": "region", "operation": "string_case", "case_mode": "upper"},
    )
    assert res_case.status_code == 200
    assert res_case.json()["success"] is True


def test_dashboard_pins_endpoint(client):
    client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})

    # Pin chart
    pin_res = client.post(
        "/api/datasets/dashboard/pins",
        json={
            "title": "Sales by Region",
            "chart_type": "Bar Chart",
            "figure_spec": {"data": [], "layout": {}},
            "source_page": "Visualizations",
            "notes": "Key revenue driver",
        },
    )
    assert pin_res.status_code == 200
    pin_item = pin_res.json()
    assert "id" in pin_item
    assert pin_item["title"] == "Sales by Region"

    # List pins
    list_res = client.get("/api/datasets/dashboard/pins")
    assert list_res.status_code == 200
    pins = list_res.json()["pinned_charts"]
    assert len(pins) >= 1

    # Delete pin
    del_res = client.delete(f"/api/datasets/dashboard/pins/{pin_item['id']}")
    assert del_res.status_code == 200


def test_ai_query_enhanced_insights(client):
    client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})

    # Execute AI Query without configured key -> reports LLM is not working right now
    res = client.post(
        "/api/query",
        json={"question": "What are the top 5 highest sales regions?"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["route"] in ["dataframe_analysis", "statistical_summary"]
    assert "insights" in data
    assert "LLM is not working right now" in data["insights"] or "Executive Summary" in data["insights"]


