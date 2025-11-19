import geocoder

g = geocoder.ip('me')
if g.ok:
    print(f"✅ Current Location: Latitude = {g.latlng[0]}, Longitude = {g.latlng[1]}")
else:
    print("❌ Unable to fetch GPS location")
