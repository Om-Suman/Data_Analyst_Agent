import io
import pytest


def test_cleaning_rejects_preview_from_another_dataset(client):
    client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})
    dataset = client.get("/api/datasets").json()["datasets"][0]
    plan = {
        "expected_dataset": dataset["name"],
        "expected_version": dataset["version"],
        "expected_uploaded_at": dataset["uploaded_at"],
    }
    assert client.post("/api/cleaning/preview", json=plan).status_code == 200
    client.post("/api/datasets/sample", json={"sample_name": "Employee Data"})
    refused = client.post("/api/cleaning/apply", json=plan)
    assert refused.status_code == 400
    assert "dataset changed" in refused.json()["detail"]
    current = next(item for item in client.get("/api/datasets").json()["datasets"] if item["is_active"])
    assert current["name"] == "Employee Data"
    assert current["version"] == 1


def test_sqlite_extension_upload(client, tmp_path):
    import sqlite3

    path = tmp_path / "table.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE sample (score INTEGER)")
        connection.execute("INSERT INTO sample VALUES (42)")
        connection.commit()
    finally:
        connection.close()
    result = client.post("/api/datasets/upload", files={"files": (path.name, path.read_bytes())})
    assert result.status_code == 200, result.json()
    assert client.get("/api/datasets/preview").json()["data"] == [{"score": 42}]


def test_connection_check_reports_unavailable_models(client, monkeypatch):
    from unittest.mock import Mock

    client.post("/api/config", json={"groq_api_key": "synthetic-key", "primary_model": "listed", "fallback_model": "missing"})
    response = Mock(status_code=200)
    response.json.return_value = {"data": [{"id": "listed", "active": True}]}
    request = Mock(return_value=response)
    monkeypatch.setattr("backend.api.routes_config.requests.get", request)
    result = client.get("/api/config/connection")
    assert result.json() == {"unavailable_models": ["missing"]}
    assert request.call_args.kwargs["headers"]["Authorization"] == "Bearer synthetic-key"


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
            "groq_api_key": "gsk_test_secret_key_12345",
            "primary_model": "qwen/qwen3.8-27b",
            "theme": "light",
            "temperature": 0.5,
        },
    )
    assert update_res.status_code == 200
    updated = update_res.json()
    assert updated["has_api_key"] is True
    assert updated["primary_model"] == "qwen/qwen3.8-27b"
    assert updated["theme"] == "light"
    assert updated["temperature"] == 0.5
    # Ensure raw API key is never exposed
    assert "gsk_test_secret_key_12345" not in updated["api_key_masked"]
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
        json={"method": "Z-Score", "threshold": 2.5, "columns": ["sales"]},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["method"] == "Z-Score"
    assert "n_anomalies" in data
    assert "figure_spec" in data
    assert data["columns_used"] == ["sales"]
    assert all("customer_id" in row for row in data["anomalous_rows"])
    client.post("/api/datasets/sample", json={"sample_name": "Employee Data"})
    assert client.get("/api/anomalies/download").status_code == 404


def test_reports_endpoints(client):
    client.post("/api/datasets/sample", json={"sample_name": "Sales Data"})

    # HTML report
    html_res = client.post(
        "/api/reports/html",
        json={"include_sample": True, "include_stats": True, "include_insights": False},
    )
    assert html_res.status_code == 200
    assert "<!DOCTYPE html>" in html_res.text
    assert "Sales Data · Version 1" in html_res.text
    assert "Data Sample (First 20 Rows)" in html_res.text
    assert "Data Sample (First 20 Rows)" not in client.post("/api/reports/html", json={}).text

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


@pytest.mark.parametrize("extension", ["txt", "pdf", "docx"])
def test_document_qa_endpoint(client, monkeypatch, extension):
    from modules import document_rag

    text_content = "The verification project deadline is September 14."
    if extension == "pdf":
        import fitz

        with fitz.open() as pdf:
            pdf.new_page().insert_text((72, 72), text_content)
            content = pdf.tobytes()
    elif extension == "docx":
        from docx import Document

        document = Document()
        document.add_table(rows=1, cols=1).cell(0, 0).text = text_content
        stream = io.BytesIO()
        document.save(stream)
        content = stream.getvalue()
    else:
        content = text_content.encode("utf-8")

    def fail_on_upload_indexing():
        pytest.fail("Upload should not initialize the embedding model")

    monkeypatch.setattr(document_rag, "llamaindex_available", fail_on_upload_indexing)
    filename = f"verification.{extension}"
    files = {"files": (filename, content, "application/octet-stream")}
    upload_res = client.post("/api/datasets/upload", files=files)
    assert upload_res.status_code == 200, upload_res.json()
    assert upload_res.json()["active_dataset"] == filename
    assert client.get("/api/datasets").json()["datasets"][0]["is_text"] is True

    monkeypatch.setattr(document_rag, "llamaindex_available", lambda: False)
    monkeypatch.setattr(document_rag, "query_llm", lambda **kwargs: ("❌ unavailable", "none"))

    qa_res = client.post(
        "/api/document/qa",
        json={"question": "What is the verification project deadline?"},
    )
    assert qa_res.status_code == 200
    data = qa_res.json()
    assert data["error"] == "AI answer unavailable"
    assert "September 14" in data["sources"][0]["text"]


def test_upload_rejects_pdf_without_extractable_text(client):
    import fitz

    with fitz.open() as pdf:
        pdf.new_page()
        content = pdf.tobytes()
    result = client.post(
        "/api/datasets/upload",
        files={"files": ("empty.pdf", content, "application/pdf")},
    )
    assert result.status_code == 400
    assert "upload a searchable copy" in result.json()["detail"]
    assert client.get("/api/datasets").json()["datasets"] == []


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


