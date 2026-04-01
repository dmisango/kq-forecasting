import gdown
import os

def download_if_missing():
    files = {
        "artifacts/arima_model.pkl":             "1qPxiNMg5QBF-2ssTJwtSfA3k4YXQAsmJ",
        "artifacts/exog_scaler.pkl":             "1chvdDPAj24PhxjhDx3R-j8MHshDt8tNV",
        "artifacts/label_encoders.pkl":          "1xbnGcRBHdnqOfR8H8VYzg8qVT49wN3ka",
        "artifacts/model_meta.pkl":              "1SYZ4XzcxYVbvzXD1r-kJL8Cx_y4nK0By",
        "artifacts/price_scaler.pkl":            "1cfoSGDP0ni_vWUDBWq2lJHjQl_19QZEb",
        "artifacts/residual_scaler.pkl":         "1POgEDDRadnEdnbmGQDdknIxSXimLczYI",
        "artifacts/hybrid_lstm_model.keras":     "10OMRP_l-2QRVWtVbW0XcU4MdyZ9tJUFG",
        "artifacts/standalone_lstm_model.keras": "19NQOfI_vHZ8mHWbiehC6gYSd0v6HIUVk",
        "bookings.xlsx":                         "1N5HoxsP3ExaKV3upt3fClg4Fxh-uJChV",
    }

    for local_path, file_id in files.items():
        if not os.path.exists(local_path):
            dir_name = os.path.dirname(local_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
                print(f"Downloading {local_path}...")
                gdown.download(
                    f"https://drive.google.com/uc?id={file_id}",
                    local_path,
                    quiet=False,
                    fuzzy=True
                )
                print(f"✓ {local_path} ready")
