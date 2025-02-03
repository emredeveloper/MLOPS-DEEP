from deepseek_mlops.data_loader import load_data

def test_load_data():
    data = load_data()
    assert data is not None
    assert "target" in data.columns
