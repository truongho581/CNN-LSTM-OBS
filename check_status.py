import requests

def check_iris_status():
    url = "https://service.iris.edu/fdsnws/station/1/"
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            print("✅ IRIS đang hoạt động bình thường.")
        else:
            print(f"⚠️ IRIS trả về mã lỗi: {response.status_code}")
    except requests.exceptions.Timeout:
        print("⏳ IRIS không phản hồi (timeout).")
    except requests.exceptions.ConnectionError:
        print("❌ Không kết nối được đến IRIS (ConnectionError).")
    except Exception as e:
        print(f"❌ Lỗi khác: {e}")

# Chạy hàm
check_iris_status()
