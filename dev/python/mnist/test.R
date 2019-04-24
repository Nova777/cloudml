#install.packages("cloudml")

library(cloudml)

#gcloud_install()

#gcloud_init()

job <- cloudml_train_python("c://projectr/cloudml/dev/python/mnist/trainer/",
                            master_type = "standard_gpu")

#TODO add run
job_collect("cloudml_2019_03_13_215824141")
