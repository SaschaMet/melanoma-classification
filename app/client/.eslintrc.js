module.exports = {
    extends: [ 'eslint:recommended', 'standard', 'plugin:react/recommended' ],
    plugins: [
        'react'
    ],
    parserOptions: {
        ecmaVersion: 2020,
        sourceType: 'module',
        ecmaFeatures: {
            jsx: true,
        },
    },
    ignorePatterns: [ '.gitignore' ],
    root: true,
    env: {
        browser: true,
        jquery: true,
        jest: true,
    },
    globals: {
        console: true,
        module: true,
        window: true,
        require: true,
        react: true,
        dataLayer: true,
        context: true,
        cy: true,
        Cypress: true,
    },
    settings: {
        react: {
            version: 'detect',
        },
    },
    /**
     * 0 = turned off
     * 1 = warning
     * 2 = error
     */
    rules: {
        'new-cap': 1,
        'no-caller': 2,
        'no-eq-null': 2,
        indent: [ 'error', 4 ],
        'no-tabs': 2,
        'linebreak-style': [ 'error', 'unix' ],
        quotes: [ 'error', 'single' ],
        semi: [ 'error', 'always' ],
        'comma-dangle': ['error', {
            arrays: 'never',
            objects: 'always',
            imports: 'never',
            exports: 'never',
            functions: 'never',
        } ],
        'one-var': 0,
        eqeqeq: [ 'error', 'smart' ],
        curly: 0,
        'for-direction': 0,
        complexity: [ 'error', 20 ], // 20 is default
        'no-undef': 2,
        'no-plusplus': 0,
        'no-underscore-dangle': 0,
        'wrap-iife': [ 'error', 'any' ],
        'no-alert': 1,
        'no-empty-function': 2,
        'no-useless-catch': 2,
        'no-eval': 2,
        'no-implied-eval': 2,
        'no-script-url': 2,
        'no-useless-call': 2,
        'vars-on-top': 0,
        'no-console': 0,
        'no-implicit-globals': 2,
        'no-return-assign': 2,
        'no-unused-expressions': 2,
        'no-unused-vars': 2,
        radix: 0,
        'no-trailing-spaces': 0,
        'array-bracket-spacing': 0,
        // es6
        'arrow-spacing': [ 'error', { before: true, after: true, } ],
        'no-confusing-arrow': [ 'error', { allowParens: false, } ],
        'arrow-parens': [ 'error', 'as-needed', { requireForBlockBody: true, } ],
        'no-useless-constructor': 2,
        'no-dupe-class-members': 2,
        'no-duplicate-imports': 2,
        'no-useless-computed-key': 2,
        'no-restricted-properties': 2,
        'operator-linebreak': 2,
        'no-nested-ternary': 2,
        'no-unneeded-ternary': 2,
        'standard/object-curly-even-spacing': [ 2, 'either' ],
        'standard/array-bracket-even-spacing': [ 2, 'either' ],
        'standard/computed-property-even-spacing': [ 2, 'even' ],
        'standard/no-callback-literal': [ 2, [ 'cb', 'callback' ] ],
        // react
        'react/prop-types': 0,
        // mui,
        'no-restricted-imports': [
            'error',
            {
                patterns: ['@material-ui/*/*/*', '!@material-ui/core/test-utils/*'],
            }
        ],
    },
};
