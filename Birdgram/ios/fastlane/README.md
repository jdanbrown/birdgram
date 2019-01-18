fastlane documentation
================
# Installation

Make sure you have the latest version of the Xcode command line tools installed:

```
xcode-select --install
```

Install _fastlane_ using
```
[sudo] gem install fastlane -NV
```
or alternatively using `brew cask install fastlane`

# Available Actions
## iOS
### ios switch
```
fastlane ios switch
```
Switch xcode settings to then given env (handled by bin/fastlane-env, noop in fastlane)
### ios create
```
fastlane ios create
```
Create all resources: bundle ids, apps, provisioning profiles, device registration, ...
### ios test-flight
```
fastlane ios test-flight
```
Build and upload to TestFlight

----

This README.md is auto-generated and will be re-generated every time [fastlane](https://fastlane.tools) is run.
More information about fastlane can be found on [fastlane.tools](https://fastlane.tools).
The documentation of fastlane can be found on [docs.fastlane.tools](https://docs.fastlane.tools).
