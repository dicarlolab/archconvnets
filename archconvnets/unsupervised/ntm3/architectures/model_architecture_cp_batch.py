add_sharpen_layer(LAYERS, RW+'_F', [RW+'_SHIFTED', RW+'_GAMMA'], init=init)
add_dotT_layer(LAYERS, 'ERASE_HEAD', ['W_F', 'ERASE'], init=init)

## above
add_sum_layer(LAYERS, 'SUM', init=init)
add_add_layer(LAYERS, 'ERR', ['SUM', -1], scalar=-1, init=init)
add_sq_points_layer(LAYERS, 'SQ_ERR', init=init)
add_sum_layer(LAYERS, 'SUM_ERR', init=init)

