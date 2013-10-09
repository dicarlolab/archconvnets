import archconvnets.dataprovider as dp
import dldata.stimulus_sets.hvm as hvm
import imagenet

def test_dataprovider_hvm():
    dataset = hvm.HvMWithDiscfade()
    imgs = dataset.get_images('float32', {'size': (128, 128), 'global_normalize': False})
    metadata = dataset.meta['category']
    provider = dp.Dldata2ConvnetProviderBase(imgs, metadata, 200)

    assert provider.get_data_dims() == 128 * 128, provider.get_data_dims()
    assert provider.batch_range == range(29), provider.batch_range

    X = provider.get_next_batch()
    X1 = provider.get_next_batch()

    assert X[0] == X1[0] == 1
    assert X[1] == 0
    assert X1[1] == 1

    assert X[2][0].shape == X1[2][0].shape == (16384, 200)


def test_dataprovider_imagenet():
    dataset = imagenet.dldatasets.PixelHardSynsets2013ChallengeTop25Screenset()
    imgs = dataset.get_images(dataset.default_preproc)
    metadata = dataset.meta['synset']
    provider = dp.Dldata2ConvnetProviderBase(imgs, metadata, 100)

    assert provider.get_data_dims() == 256 * 256 * 3, provider.get_data_dims()

    X = provider.get_next_batch()
    X1 = provider.get_next_batch()

    assert X[0] == X1[0] == 1
    assert X[1] == 0
    assert X1[1] == 1

    assert X[2][0].shape == X1[2][0].shape == (256 * 256 * 3, 100)



def test_dataprovider_hvm_allbatches():
    dataset = hvm.HvMWithDiscfade()
    imgs = dataset.get_images('float32', {'size': (128, 128), 'global_normalize': False})
    metadata = dataset.meta['category']
    provider = dp.Dldata2ConvnetProviderBase(imgs, metadata, 200, batch_range=[0, 15])

    for i in range(3):
        X = provider.get_next_batch()

    assert X[0] == 2, 'epoch should be %d but is %d' % (2, X[0])
    assert X[1] == 0, 'batch_num should be %d but is %d' % (0, X[1])
