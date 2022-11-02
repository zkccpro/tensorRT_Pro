#!/bin/bash

echo Remove src/tensorRT/amirstan_plugin
rm -rf src/tensorRT/amirstan_plugin
mkdir src/tensorRT/amirstan_plugin

echo Copy [amirstan_plugin/src] to [src/tensorRT/amirstan_plugin/src]
cp -r amirstan_plugin/src src/tensorRT/amirstan_plugin/src

echo Copy [amirstan_plugin/include] to [src/tensorRT/amirstan_plugin/include]
cp -r amirstan_plugin/include src/tensorRT/amirstan_plugin/include

echo Configure your tensorRT path to amirstan_plugin
echo After that, you can execute the command 'make -j64'