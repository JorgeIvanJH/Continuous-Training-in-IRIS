#!/bin/bash
set -e

iris session IRIS <<'EOF'

Write "STARTING IRIS CONFIGURATION...",!

Write "Installing IPM...",!
s version="latest" s r=##class(%Net.HttpRequest).%New(),r.Server="pm.community.intersystems.com",r.SSLConfiguration="ISC.FeatureTracker.SSL.Config" d r.Get("/packages/zpm/"_version_"/installer"),$system.OBJ.LoadStream(r.HttpResponse.Data,"c")

Write "Installing IPM packages...",!
zpm
repo -r -n registry -url https://pm.community.intersystems.com/ -user "" -pass ""
install csvgenpy
quit

Write "Importing ObjectScript packages...",!
Do $system.OBJ.Import("/usr/irissys/mgr/MLpipeline", "ck")

Write "Loading CSV data to IRIS...",!
SET exists = ##class(%SYSTEM.SQL.Schema).TableExists("MLpipeline.PointSamples")
IF 'exists { do ##class(MLpipeline.DataManager).UploadCSVtoIRIS("/dur/data/point_samples.csv","USER", "MLpipeline","PointSamples") }

Write "Enabling analytics...",!
do EnableDeepSee^%SYS.cspServer("/csp/user/")

Write "Disabling password expiration...",!
ZN "%SYS"
Do ##class(Security.Users).UnExpireUserPasswords("*")

Write "IRIS CONFIGURATION COMPLETED.",!
halt
EOF
