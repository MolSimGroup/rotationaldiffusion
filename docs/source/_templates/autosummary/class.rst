{%- set module_prefixes = {
    'RotationalDiffusion.quaternions': 'qops',
} -%}
{%- set default_prefix = 'rd' -%}
{%- set prefix = module_prefixes.get(module, default_prefix) -%}
{%- set display_name = '`' + prefix + '.' + objname + ('()' if objtype in ['function', 'method'] else '') + '`' -%}
{{ display_name | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: parallelizable, get_supported_backends