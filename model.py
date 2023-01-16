from layers import *

def create_model(im_size, stem_patch_size=2, window_size=8, n_blocks=[2, 2, 4, 2], depths=[64, 128, 256, 512], decode_dim=256, n_labels=1):
    attn_cnt = 0
    features = []
    resize_shape = (im_size//4,im_size//4)
    out_shape = (im_size, im_size)

    inputs = Input((im_size,im_size,3))

    x = Conv2D(32, kernel_size=stem_patch_size, strides=stem_patch_size, use_bias=True, name="stem_conv")(inputs)

    for iblock in range(len(n_blocks)):
        x = Conv2D(depths[iblock], kernel_size=3, strides=2, use_bias=False, padding="same")(x)

        x_attn = x
        x_conv = x

        for i in range(n_blocks[iblock]):
            x_attn = swin_transformer_block(x_attn, window_size, name=f"attn_{attn_cnt}_"); attn_cnt += 1
            x_conv = mscan(x_conv)

            x_attn = Dropout(0.01)(x_attn)
            x_conv = Dropout(0.01)(x_conv)

        x_merge = Concatenate()([x_attn, x_conv])
        x_merge = Conv2D(x_attn.shape[-1], 1, padding="same")(x_merge)

        x_attn_mul = Conv2D(x_attn.shape[-1], 1, padding="same", activation="sigmoid")(x_attn)
        x_conv_mul = Conv2D(x_conv.shape[-1], 1, padding="same", activation="sigmoid")(x_conv)

        x_merge_mul = Multiply()([x_merge, x_attn_mul, x_conv_mul])
        x = Add()([x_merge, x_merge_mul])

        features.append(x)

    resize_features = []

    for feature in features:
        x = Conv2D(decode_dim, 1, padding="same")(feature)
        x = local_emphasis(x, decode_dim, resize_shape)
        resize_features.append(x)

    resize_features = resize_features[::-1] # smaller to bigger
    x = resize_features[0]

    for resize_feature in resize_features[1:]:
        x = Concatenate()([x, resize_feature])
        x = Conv2D(decode_dim, 1, padding="same")(x)
        x = extract_block(x, decode_dim)

    x = tf.image.resize(x, out_shape)
    x = extract_block(x, decode_dim)

    x = Conv2D(n_labels, 1, padding="same", activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)
    
    return model

if __name__ == "__main__":
    import os
    from utils import *

    os.environ["CUDA_VISIBLE_DEVICES"]=""

    settings = get_settings()
    globals().update(settings)

    n_labels = 1

    model = create_model(im_size, stem_patch_size, window_size, n_blocks, depths, decode_dim, n_labels)

    model.summary()

    print(model.output)

    inp = tf.ones((1, im_size, im_size, 3))
    out = model.predict(inp)