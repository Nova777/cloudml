#install.packages("cloudml")

#library(cloudml)

#gcloud_install()

#gcloud_init()

job <- cloudml_train_python("c://projectr/cloudml/dev/python/babyweight/trainer/",
                            master_type = "standard_gpu")

