#!/bin/sh

# set scratch.py decode_spec.py encode_spec.py prepare_data.py

# for i in $@; do
	# git checkout master $i
# done

for i in *.py; do
	git checkout master $i
done

