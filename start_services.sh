#!/bin/bash

systemctl start speech-dispatcher
pulseaudio --kill
pulseaudio --start

