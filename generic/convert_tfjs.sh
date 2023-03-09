
echo
echo "Converting, no compression..."
tensorflowjs_converter \
	--control_flow_v2=True \
	--input_format=tf_saved_model \
	--metadata= \
	--saved_model_tags=serve \
	--signature_name=serving_default \
	--strip_debug_ops=True \
	--weight_shard_size_bytes=4194304 \
	${1} ${1}/tfjs
du -h tfjs

echo
echo "Converting, float16 compression..."
tensorflowjs_converter \
	--control_flow_v2=True \
	--input_format=tf_saved_model \
	--metadata= \
	--quantize_float16=* \
	--saved_model_tags=serve \
	--signature_name=serving_default \
	--strip_debug_ops=True \
	--weight_shard_size_bytes=4194304 \
	${1} ${1}/tfjs_float16
du -h ${1}/tfjs_float16

echo
echo "Converting, uint16 compression..."
tensorflowjs_converter \
	--control_flow_v2=True \
	--input_format=tf_saved_model \
	--metadata= \
	--quantize_uint16=* \
	--saved_model_tags=serve \
	--signature_name=serving_default \
	--strip_debug_ops=True \
	--weight_shard_size_bytes=4194304 \
	${1} ${1}/tfjs_uint16
du -h ${1}/tfjs_uint16
