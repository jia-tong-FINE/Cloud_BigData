# Generated by Django 3.1.3 on 2020-11-13 16:04

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('movie_rcmd', '0004_auto_20201113_2355'),
    ]

    operations = [
        migrations.RenameField(
            model_name='movie',
            old_name='avg_rating',
            new_name='mean_rating',
        ),
    ]
