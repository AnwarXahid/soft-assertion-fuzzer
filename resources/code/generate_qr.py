import qrcode

# URL that never expires
url = "https://github.com/ISU-PAAL/soft-assertion-fuzzer"

# Create QR code
qr_img = qrcode.make(url)

# Save the QR code as a PNG file
qr_img.save("../images/soft_assertion_fuzzer_qr.png")

print("QR code created successfully!")
