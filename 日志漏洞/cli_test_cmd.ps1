Invoke-RestMethod -Method POST -Uri http://127.0.0.1:5000/login `
  -ContentType 'application/json' `
  -Body '{"username":"alice","password":"SuperSecret123","otp":"123456"}'