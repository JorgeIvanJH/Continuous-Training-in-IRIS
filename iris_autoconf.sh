#!/bin/bash
set -e

iris session IRIS <<'EOF'

/* Install IPM/ZPM client if you still need that first
   (your original snippet did this already) */
s version="latest" s r=##class(%Net.HttpRequest).%New(),r.Server="pm.community.intersystems.com",r.SSLConfiguration="ISC.FeatureTracker.SSL.Config" d r.Get("/packages/zpm/"_version_"/installer"),$system.OBJ.LoadStream(r.HttpResponse.Data,"c")

/* Configure registry */
zpm
repo -r -n registry -url https://pm.community.intersystems.com/ -user "" -pass ""
install csvgenpy
quit

/* Import and Compile the MLpipeline package*/
/* The "ck" flags will Compile and Keep the source */
Do $system.OBJ.Import("/usr/irissys/mgr/MLpipeline", "ck")

/* Upload csv data ONCE to Table Automatically */
# SET exists = ##class(%SYSTEM.SQL.Schema).TableExists("MLpipeline.PointSamples") # TODO: change to upload keeping data types
# IF 'exists {   do ##class(MLpipeline.DataManager).UploadCSVtoIRIS("/dur/data/point_samples.csv","USER", "MLpipeline","PointSamples")   }

/* Enable Analytics */
do EnableDeepSee^%SYS.cspServer("/csp/user/") # Enable for USER namespace


halt
EOF
